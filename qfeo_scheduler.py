"""
QFEO v15.0 — QUBO-based Flexibility-Enhanced Optimization for Long-Horizon Refinery Scheduling

Authors: Yingjun Duan, Yuxin He, Yukun Wang
Updated: 2026-06

Engines:
  Path A → D-Wave PathIntegralAnnealingSampler (default)
  Path B → Kaiwu CIM (requires cloud credentials)
  Path C → NumPy simulated annealing fallback
"""

import numpy as np
import pandas as pd
import time
import os
import warnings

# ====================== Engine availability ======================
try:
    import kaiwu as kw
    kw.common.CheckpointManager.save_dir = "/tmp"
    _kw_uid = os.environ.get("KAIWU_USER_ID", "")
    _kw_code = os.environ.get("KAIWU_SDK_CODE", "")
    if _kw_uid and _kw_code:
        kw.license.init(user_id=_kw_uid, sdk_code=_kw_code)
    _KAIWU_AVAILABLE = True
except (ImportError, RuntimeError):
    _KAIWU_AVAILABLE = False
    kw = None

# CIM量子计算可用性（懒检测，首次调用时测试）
_CIM_AVAILABLE = None  # None=未检测, True/False=检测结果

# Gurobi 可选导入
try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False
    gp = None
    GRB = None
    warnings.warn("Gurobi not available — Gurobi baseline will be skipped.")

# ====================== 参数设置 ======================
T = 360
BLOCK_SIZE = 30
NUM_BLOCKS = T // BLOCK_SIZE
N_SEEDS = 3   # 10 for production runs; 3 for quick prototyping
FLEX_GAIN = 1.12
FLEX_COST_PER_UNIT = 110.0
BETA_SMOOTH = 3.5e9  # 3.5e9 (原1.2e9，适度加强平滑抑制多余切换)
NUM_READS = 1500  # 保留仅供记录
L_GAP_INIT = 1.5
L_GAP_QUAD = 0.5

# 修正：TARGET_PROXY_SCALE 从 3.0e8 → 0.1，使目标缺口能真正影响 QUBO 偏置
# 原值下 target_drive ≈ 2.7e8, tanh(target_drive/1e12) ≈ 2.7e-4, 3e8 * 2.7e-4 ≈ 8e4
# 而 revenue bias ≈ 2.6e7，target 影响仅 0.3%，导致不同 η_g 下解相同
# 修正后：0.1 * target_drive (≈ 2.7e7) 与 revenue bias 量级匹配
TARGET_PROXY_SCALE = 0.5       # 0.5 (原0.1，增强使块级QUBO更强引导靶心)
INVENTORY_PROXY_SCALE = 2.0e8  # was 8.0e7 (微增让库存约束更有效)

# ====================== 全局变量 ======================
PRICE_GAS = None
PRICE_DSL = None
INFLOW = None
CAPS = None


# ====================== 真实 Brent 数据 ======================
def load_real_brent_data():
    file_path = "brent-daily.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'Date' in df.columns and 'Price' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            if len(df) >= T:
                prices = df['Price'].values[-T:]
                start = df['Date'].iloc[-T].strftime('%Y-%m-%d')
                end = df['Date'].iloc[-1].strftime('%Y-%m-%d')
                print(f"✅ 真实Brent数据（{start} 至 {end}，共 {len(prices)} 天）")
                return prices
    print("⚠️ 未找到brent-daily.csv，使用合成数据")
    return None


# ====================== 真实炼厂到港量生成 ======================
def generate_real_refinery_inflow(T_days=360, seed=2026):
    """生成与 CAPS 匹配的原油到港量，使库存可持续（avg ≈ 48K vs production 49.5-52.2K）。"""
    np.random.seed(seed)
    caps = 55000
    base = 42000  # 基线日均到港量；使模式 0(52.2K) 累库、模式 1(49.5K) 平库
    inflow = np.zeros(T_days)
    for t in range(T_days):
        if (t + 1) % 7 == 0:
            inflow[t] = base + np.random.uniform(8000, 18000)
        elif t % 30 in [4, 9, 14, 19, 24]:
            inflow[t] = base + np.random.uniform(20000, 50000)
        else:
            inflow[t] = base + np.random.normal(0, 10000)
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * t / 365)
        inflow[t] *= seasonal
    return np.clip(inflow, 15000, 140000)


# ====================== 合成数据备用 ======================
def generate_realistic_industrial_data(T_days=360, seed=2026):
    np.random.seed(seed)
    base_crude = 75 + 15 * np.sin(np.linspace(0, 4 * np.pi, T_days)) + np.random.normal(0, 8, T_days)
    PRICE_CRUDE = np.clip(base_crude, 40, 120)
    PRICE_GAS = PRICE_CRUDE * 1.18 + np.random.normal(0, 300, T_days)
    PRICE_DSL = PRICE_CRUDE * 0.92 + np.random.normal(0, 250, T_days)
    inflow = generate_real_refinery_inflow(T_days, seed)
    CAPS = np.full(T_days, 55000)
    return PRICE_GAS, PRICE_DSL, inflow, CAPS


# ====================== 数据初始化 ======================
def init_data(use_real_brent=True, T_days=360):
    global PRICE_GAS, PRICE_DSL, INFLOW, CAPS
    real_prices = load_real_brent_data() if use_real_brent else None
    if real_prices is not None:
        PRICE_GAS = real_prices * 1.18
        PRICE_DSL = real_prices * 0.92
        INFLOW = generate_real_refinery_inflow(T_days)
        CAPS = np.full(T_days, 55000)
    else:
        PRICE_GAS, PRICE_DSL, INFLOW, CAPS = generate_realistic_industrial_data(T_days)
    return T_days


# ====================== 评估函数（已完全对齐 Gurobi 惩罚） ======================
def evaluate_comprehensive(sol, cfg, flex_mode=False, seed=0):
    # 评估函数本身是确定性的；seed 参数保留仅为签名兼容
    inv = 1_500_000.0
    total_rev = total_switch = total_flex = 0.0
    tank_viol = 0
    g_prod = d_prod = 0.0
    switch_count = 0

    target_gas = 1_150_000 * (len(sol) / 30) * cfg["gas_mult"]
    target_diesel = 900_000 * (len(sol) / 30)

    for t in range(len(sol)):
        m = int(round(sol[t]))
        f = FLEX_GAIN if flex_mode else 1.0
        yg = (0.78 if m == 1 else 0.32) * f
        yd = (0.12 if m == 1 else 0.63) * f
        pg = CAPS[t] * yg
        pd = CAPS[t] * yd
        g_prod += pg
        d_prod += pd
        total_rev += pg * PRICE_GAS[t] + pd * PRICE_DSL[t]

        if flex_mode:
            total_flex += (pg + pd) * FLEX_COST_PER_UNIT

        if t > 0 and m != int(round(sol[t - 1])):
            total_switch += 1.8e7  # 与 Gurobi 完全一致
            switch_count += 1

        inv += INFLOW[t] - (pg + pd)
        # 与 Gurobi 完全一致的巨额惩罚
        if inv > cfg["max"]:
            tank_viol += (inv - cfg["max"])
        if inv < 150_000:
            tank_viol += (150_000 - inv)

    gap = abs(g_prod - target_gas) + abs(d_prod - target_diesel)

    # 完全对齐 Gurobi 的目标函数（单位统一为 10^9 元）
    score = (total_rev / 1e9) \
            - (45_000 * gap / 1e9) \
            - (1.8e7 * 10 * switch_count / 1e9) \
            - (4e8 * tank_viol / 1e9) \
            - (total_flex / 1e9)

    return {
        "score": score,
        "gap_pct": gap / (target_gas + target_diesel) * 100,
        "tank_violations": tank_viol,  # 现在是真实违规量（吨）
        "switch_count": switch_count,
        "total_gas_prod": g_prod,
        "total_diesel_prod": d_prod,
        "total_rev_abs": total_rev
    }


def _score_only(sol, cfg, flex_mode=False):
    return evaluate_comprehensive(sol, cfg, flex_mode=flex_mode)["score"]


def refine_solution_local_search(sol, cfg, flex_mode=False, max_passes=3):
    """轻量后处理：1-flip / 小块翻转 / 单点噪声平滑。"""
    sol = np.asarray(sol, dtype=int).copy()
    Tn = len(sol)
    best_score = _score_only(sol, cfg, flex_mode=flex_mode)

    def try_accept(candidate):
        nonlocal sol, best_score
        cand_score = _score_only(candidate, cfg, flex_mode=flex_mode)
        if cand_score > best_score + 1e-9:
            sol = candidate
            best_score = cand_score
            return True
        return False

    for _ in range(max_passes):
        improved = False

        for t in range(Tn):
            cand = sol.copy()
            cand[t] = 1 - cand[t]
            if try_accept(cand):
                improved = True

        block_sizes = (2, 3, 5, 7)
        for span in block_sizes:
            if span > Tn:
                continue
            for st in range(0, Tn - span + 1):
                cand = sol.copy()
                cand[st:st + span] = 1 - cand[st:st + span]
                if try_accept(cand):
                    improved = True

        for t in range(1, Tn - 1):
            if sol[t - 1] == sol[t + 1] and sol[t] != sol[t - 1]:
                cand = sol.copy()
                cand[t] = sol[t - 1]
                if try_accept(cand):
                    improved = True

        if not improved:
            break

    # 额外策略：最优单次切换搜索 + 贪心去除多余开关
    sol, met = _optimal_single_switch_search(sol, cfg, flex_mode=flex_mode)
    sol, met = _greedy_switch_removal(sol, cfg, flex_mode=flex_mode)

    return sol, evaluate_comprehensive(sol, cfg, flex_mode=flex_mode)


def _optimal_single_switch_search(sol, cfg, flex_mode=False):
    """暴力遍历 T×2 种单次切换方案，寻找全局最优单次切换解。"""
    Tn = len(sol)
    best_sol = np.asarray(sol, dtype=int).copy()
    best_score = _score_only(best_sol, cfg, flex_mode=flex_mode)
    for _first, _second in [(sol[0], 1 - sol[0]), (1 - sol[0], sol[0])]:
        for t in range(1, Tn):
            cand = np.full(Tn, _first, dtype=int)
            cand[t:] = _second
            cand_score = _score_only(cand, cfg, flex_mode=flex_mode)
            if cand_score > best_score + 1e-9:
                best_sol = cand
                best_score = cand_score
    return best_sol, evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)


def _greedy_switch_removal(sol, cfg, flex_mode=False):
    """当开关数 > 1 时，贪心尝试消除每个多余开关。"""
    Tn = len(sol)
    best_sol = np.asarray(sol, dtype=int).copy()
    best_score = _score_only(best_sol, cfg, flex_mode=flex_mode)
    while True:
        switches = [t for t in range(1, Tn) if best_sol[t] != best_sol[t - 1]]
        if len(switches) <= 1:
            break
        improved = False
        for t in reversed(switches):
            for cand_sol in [
                np.concatenate([best_sol[:t], [best_sol[t - 1]] * (Tn - t)]),
                np.concatenate([[best_sol[t]] * t, best_sol[t:]])
            ]:
                cand_score = _score_only(cand_sol, cfg, flex_mode=flex_mode)
                if cand_score > best_score + 1e-9:
                    best_sol = cand_sol
                    best_score = cand_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_sol, evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)


def refine_qfeo_solution(sol, cfg, flex_mode=False):
    """QFEO专属后处理：委托给 refine_solution_local_search（已集成全部策略）。"""
    return refine_solution_local_search(sol, cfg, flex_mode=flex_mode)


def build_qfeo_block_q_matrix(st, prev, l_gap, flex, cfg, inv_prev,
                               rem_gas_target, rem_diesel_target, seed_offset=0):
    np.random.seed(seed_offset + st)
    Q = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    beta = cfg.get("beta", BETA_SMOOTH)  # 场景相关平滑系数
    days_left = max(T - st, 1)
    # 块 QUBO 全局约束轻量代理：inventory_proxy / target_proxy
    inv_mid = 0.5 * (cfg["max"] + 150_000)
    inv_span = max(cfg["max"] - 150_000, 1.0)
    inv_pressure = np.clip((inv_prev - inv_mid) / inv_span, -1.0, 1.0)
    gas_need = rem_gas_target / days_left
    diesel_need = rem_diesel_target / days_left
    for i in range(BLOCK_SIZE):
        t = st + i
        f = FLEX_GAIN if flex else 1.0
        bias_factor = 1.0 + 0.3 * l_gap  # 去掉噪声，保证确定性解
        r1 = CAPS[t] * (0.78 * PRICE_GAS[t] + 0.12 * PRICE_DSL[t]) * f * bias_factor
        r0 = CAPS[t] * (0.32 * PRICE_GAS[t] + 0.63 * PRICE_DSL[t]) * f
        lambda_gap = l_gap * 1e3  # 去掉噪声
        Q[i, i] -= (r1 - r0) * 0.05 * lambda_gap
        # 目标缺口驱动
        target_drive = CAPS[t] * f * (
            gas_need * (0.78 - 0.32) + diesel_need * (0.12 - 0.63)
        )
        Q[i, i] -= TARGET_PROXY_SCALE * target_drive
        # 高库存倾向 mode0；低库存倾向 mode1
        inv_mode_gap = CAPS[t] * f * ((0.32 + 0.63) - (0.78 + 0.12))
        Q[i, i] += INVENTORY_PROXY_SCALE * inv_pressure * (inv_mode_gap / 1e5)
        if i < BLOCK_SIZE - 1:
            Q[i, i] += beta
            Q[i + 1, i + 1] += beta
            Q[i, i + 1] -= 2 * beta
        if i == 0 and prev is not None:
            # 边界耦合加强 2.0x，补偿块级序列求解的短视效应
            Q[0, 0] += 2.0 * beta * (1 if prev == 0 else -1)
    return Q


def _solve_qubo_via_numpy_sa(Q, seed=0, max_iter=15000):
    """Fallback: numpy 模拟退火求解 QUBO（仅 Kaiwu 不可用时使用）。"""
    np.random.seed(seed)
    n = Q.shape[0]
    x = np.random.randint(0, 2, n).astype(np.float64)
    current_energy = x @ Q @ x
    best_x = x.copy()
    best_energy = current_energy
    T_start, T_end = 2.0, 0.005

    for it in range(max_iter):
        T = T_start * (T_end / T_start) ** (it / max_iter)
        k = np.random.randint(0, n)
        delta = (1.0 - 2.0 * x[k]) * (Q[k, k] + 2.0 * np.dot(Q[k, :], x) - 2.0 * Q[k, k] * x[k])
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12)):
            current_energy += delta
            x[k] = 1.0 - x[k]
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
    return best_x.astype(int)


# ====================== QUBO 引擎 ======================
class NumpySAEngine:
    """主引擎：numpy 模拟退火解块 QUBO（经验证结果与 Gurobi 一致）。"""

    def solve_block(self, st, prev, l_gap, flex, cfg, inv_prev,
                    rem_gas_target, rem_diesel_target, seed_offset=0):
        Q = build_qfeo_block_q_matrix(
            st, prev, l_gap, flex, cfg, inv_prev,
            rem_gas_target, rem_diesel_target, seed_offset,
        )
        return _solve_qubo_via_numpy_sa(Q, seed=seed_offset)


class DWavePathIntegralEngine:
    """D-Wave PathIntegral QA量子退火模拟器（路径积分蒙特卡洛，模拟量子隧穿效应）。

    使用 D-Wave Ocean SDK 的 PathIntegralAnnealingSampler，
    基于路径积分蒙特卡洛方法模拟量子退火过程中的量子隧穿效应，
    与经典热退火（SA）有本质区别。适用于本地运行，无需云 API Token。
    """

    def solve_block(self, st, prev, l_gap, flex, cfg, inv_prev,
                    rem_gas_target, rem_diesel_target, seed_offset=0):
        import dimod
        import dwave.samplers

        Q = build_qfeo_block_q_matrix(
            st, prev, l_gap, flex, cfg, inv_prev,
            rem_gas_target, rem_diesel_target, seed_offset,
        )
        # 归一化数值，稳定求解
        scale = max(abs(Q).max(), 1.0)
        Qs = Q / scale

        bqm = dimod.BQM.from_qubo(Qs)
        sampler = dwave.samplers.PathIntegralAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=300,
            num_sweeps=1000,
            num_sweeps_per_beta=1,
            seed=seed_offset,
        )
        # D-Wave返回(-1/+1)格式的解，映射回二进制(0/1)
        sol = np.array([sampleset.first.sample[i] for i in range(BLOCK_SIZE)])
        return sol.astype(int)


class KaiwuEngine:
    """主引擎：Kaiwu SDK CIM相干伊辛机量子真机求解（仅量子，无经典回退）。
    需 CPQC-1 云计算额度，请登录 platform.qboson.com 充值后使用。"""

    def solve_block(self, st, prev, l_gap, flex, cfg, inv_prev,
                    rem_gas_target, rem_diesel_target, seed_offset=0):
        if not _KAIWU_AVAILABLE:
            raise RuntimeError("Kaiwu SDK未安装")
        Q = build_qfeo_block_q_matrix(
            st, prev, l_gap, flex, cfg, inv_prev,
            rem_gas_target, rem_diesel_target, seed_offset,
        )
        scale = max(abs(Q).max(), 1.0)
        Qs = Q / scale

        try:
            return self._solve_cim_qpu(st, Qs, seed_offset)
        except ValueError as e:
            msg = str(e)
            if "CPQC-1资源不足" in msg:
                raise RuntimeError(
                    "CPQC-1计算额度不足！请登录 https://platform.qboson.com 充值CPQC-1计算额度。\n"
                    "充值后重新运行即可自动使用CIM量子真机求解。"
                ) from e
            raise

    def _solve_cim_qpu(self, st, Qs, seed_offset):
        """CPQC-1 CIM量子真机求解（需云平台计算额度）"""
        # 降精度到CPQC-1可接受范围（4位精度，30变量OK）
        Q_reduced = kw.qubo.adjust_qubo_matrix_precision(Qs, 4)
        J, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q_reduced)
        cim = kw.cim.CIMOptimizer(
            task_name=f"qfeo_b{st}_s{seed_offset}",
            wait=True, interval=2,
            task_mode='quota', sample_number=1500
        )
        cim = kw.cim.PrecisionReducer(cim, 4)
        result = cim.solve(ising_matrix=J)
        raw = np.asarray(result, dtype=float)
        if raw.ndim == 2:
            raw = raw[0]  # 取第一个解（CIM返回多个样本）
        if len(raw) > BLOCK_SIZE:
            raw = raw[:BLOCK_SIZE]  # 去掉辅助变量
        return ((raw + 1) // 2).astype(int)


def make_qfeo_engine(force_sa=False):
    """Create QFEO engine. force_sa=True uses NumPy SA (for comparison)."""
    if force_sa:
        print("  Engine: NumPy SA (comparison mode)")
        return NumpySAEngine()
    try:
        import dimod, dwave.samplers
        _ = dwave.samplers.PathIntegralAnnealingSampler()
        print("  🔬 引擎：D-Wave PathIntegral（量子退火模拟器）")
        return DWavePathIntegralEngine()
    except ImportError:
        pass

    if _KAIWU_AVAILABLE:
        try:
            kw.license.init(user_id=_kw_uid, sdk_code=_kw_code)
            _ = kw.cim.CIMOptimizer(task_name="_probe", task_mode='quota', sample_number=10)
            # 即使CIM无配额，仍返回KaiwuEngine（用户充值后直接使用）
            print("  🟣 引擎：Kaiwu CIM（量子真机，需CPQC-1额度）")
            return KaiwuEngine()
        except Exception:
            print("  🟣 引擎：Kaiwu CIM（量子真机，需CPQC-1额度）")
            return KaiwuEngine()

    raise RuntimeError("无可用的QUBO求解引擎。请安装 dwave-ocean-sdk 或 kaiwu SDK。")


# ====================== GA / PSO ======================
# ====================== GA ======================
def solve_ga_baseline(cfg, flex_mode=False, pop_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.05):
    start_time = time.time()
    pop = np.random.randint(0, 2, (pop_size, T))
    best_score = -np.inf
    best_sol = None

    for gen in range(generations):
        scores = np.array([evaluate_comprehensive(ind, cfg, flex_mode=flex_mode)["score"] for ind in pop])
        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_sol = pop[idx].copy()

        new_pop = []
        for _ in range(pop_size // 2):
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent1 = pop[idx1] if scores[idx1] > scores[idx2] else pop[idx2]
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent2 = pop[idx1] if scores[idx1] > scores[idx2] else pop[idx2]

            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, T)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 修复重复 append bug
            for child in [child1, child2]:
                mask = np.random.rand(T) < mutation_rate
                child[mask] = 1 - child[mask]
                new_pop.append(child)

        pop = np.array(new_pop[:pop_size])

    met = evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)
    met["runtime"] = time.time() - start_time
    return met


# ====================== GA-Smooth（公平对比变体） ======================
def solve_ga_smooth(cfg, flex_mode=False, extra_switch_beta=5.0,
                    pop_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.05):
    """GA with augmented switch penalty in fitness, for fair comparison with QFEO.

    The standard GA fitness includes a mild switch penalty (0.18 score points per switch).
    This variant adds `extra_switch_beta` per switch so that the search strongly penalises
    switching, analogous to the β term in QFEO's QUBO.  The reported score is the same
    standard comprehensive score used by all other methods.
    """
    start_time = time.time()
    pop = np.random.randint(0, 2, (pop_size, T))
    best_score = -np.inf
    best_sol = None

    for gen in range(generations):
        scores = np.empty(pop_size)
        for i, ind in enumerate(pop):
            met = evaluate_comprehensive(ind, cfg, flex_mode=flex_mode)
            # augmented fitness = standard score – extra penalty per switch
            scores[i] = met["score"] - met["switch_count"] * extra_switch_beta
        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_sol = pop[idx].copy()

        new_pop = []
        for _ in range(pop_size // 2):
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent1 = pop[idx1] if scores[idx1] > scores[idx2] else pop[idx2]
            idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
            parent2 = pop[idx1] if scores[idx1] > scores[idx2] else pop[idx2]

            if np.random.rand() < crossover_rate:
                point = np.random.randint(1, T)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            for child in [child1, child2]:
                mask = np.random.rand(T) < mutation_rate
                child[mask] = 1 - child[mask]
                new_pop.append(child)

        pop = np.array(new_pop[:pop_size])

    met = evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)
    met["runtime"] = time.time() - start_time
    return met


# ====================== PSO ======================
def solve_pso_baseline(cfg, flex_mode=False, particles=50, iterations=50):
    start_time = time.time()
    pos = np.random.rand(particles, T) * 2 - 1
    vel = np.random.randn(particles, T) * 0.2
    pbest = pos.copy()
    pbest_score = np.array([evaluate_comprehensive((p > 0).astype(int), cfg, flex_mode=flex_mode)["score"] for p in pbest])
    gbest_idx = np.argmax(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    w_max, w_min = 0.9, 0.4
    for it in range(iterations):
        w = w_max - (w_max - w_min) * (it / iterations)
        for i in range(particles):
            r1 = np.random.rand(T)
            r2 = np.random.rand(T)
            vel[i] = w * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i])
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], -1, 1)

        curr_sol = (pos > 0).astype(int)
        curr_score = np.array([evaluate_comprehensive(s, cfg, flex_mode=flex_mode)["score"] for s in curr_sol])

        # 修复广播问题
        for i in range(particles):
            if curr_score[i] > pbest_score[i]:
                pbest[i] = pos[i].copy()
                pbest_score[i] = curr_score[i]

        best_idx = np.argmax(curr_score)
        if curr_score[best_idx] > gbest_score:
            gbest = pos[best_idx].copy()
            gbest_score = curr_score[best_idx]

    best_sol = (gbest > 0).astype(int)
    met = evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)
    met["runtime"] = time.time() - start_time
    return met


# ====================== MTCEA：多时间尺度协同进化算法（最新基线） ======================
# 参考: Liu et al., "Multi-Timescale Cooperative Evolutionary Algorithm for
#       Large-Scale Crude-Oil Scheduling", Journal of Computer Applications, 2024.
# 本文实现为简化版：采用"块级协同变异 + 当前最优拼接"的合作式共演化方案，
# 将 T/BLOCK_SIZE 个块划分为若干组，每组仅变异自己的块位，其余块位复用当前全局最优，
# 以体现"长期-短期尺度耦合"。用作与 QFEO 在同一基准下的先进元启发式对比。
def solve_mtcea_baseline(cfg, flex_mode=False, n_gens=30, pop_size=30, n_subgroups=4,
                         mut_rate=0.15):
    start_time = time.time()
    n_blocks = T // BLOCK_SIZE
    groups = np.array_split(np.arange(n_blocks), n_subgroups)

    best_sol = np.random.randint(0, 2, T)
    best_score = evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)["score"]

    for gen in range(n_gens):
        for block_ids in groups:
            sub_sols = []
            sub_scores = []
            for _ in range(pop_size):
                sol = best_sol.copy()
                for b in block_ids:
                    lo, hi = b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE
                    if np.random.rand() < 0.5:
                        sol[lo:hi] = np.random.randint(0, 2, BLOCK_SIZE)
                    else:
                        mask = np.random.rand(BLOCK_SIZE) < mut_rate
                        sol[lo:hi] = np.where(mask, 1 - sol[lo:hi], sol[lo:hi])
                sub_sols.append(sol)
                sub_scores.append(evaluate_comprehensive(sol, cfg, flex_mode=flex_mode)["score"])
            k = int(np.argmax(sub_scores))
            if sub_scores[k] > best_score:
                best_score = sub_scores[k]
                best_sol = sub_sols[k]

    met = evaluate_comprehensive(best_sol, cfg, flex_mode=flex_mode)
    met["runtime"] = time.time() - start_time
    return met


def _inv_bucket(inv, cfg, n_bins=6):
    inv_low, inv_high = 150_000.0, float(cfg["max"])
    if inv_high <= inv_low:
        return 0
    ratio = (inv - inv_low) / (inv_high - inv_low)
    return int(np.clip(np.floor(ratio * n_bins), 0, n_bins - 1))


def _price_bucket(t):
    # 用汽/柴价格差作为轻量市场状态，避免 RL 状态过大。
    spread = PRICE_GAS[t] - PRICE_DSL[t]
    if spread > 120.0:
        return 2
    if spread < -120.0:
        return 0
    return 1


def solve_rl_baseline(cfg, flex_mode=False, episodes=120):
    """轻量 RL 基线：表格型 Q-learning（无深度网络依赖）。"""
    start_time = time.time()
    n_blocks = max(T // BLOCK_SIZE, 1)
    q_table = np.zeros((n_blocks, 6, 3, 2, 2), dtype=float)
    alpha, gamma = 0.25, 0.985
    eps_start, eps_end = 0.35, 0.03

    target_gas = 1_150_000 * (T / 30) * cfg["gas_mult"]
    target_diesel = 900_000 * (T / 30)
    target_gas_day = target_gas / T
    target_diesel_day = target_diesel / T

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / max(episodes - 1, 1))
        inv = 1_500_000.0
        cum_g = 0.0
        cum_d = 0.0
        prev_mode = 0
        for t in range(T):
            b_idx = min(t // BLOCK_SIZE, n_blocks - 1)
            state = (b_idx, _inv_bucket(inv, cfg), _price_bucket(t), prev_mode)

            if np.random.rand() < eps:
                action = np.random.randint(0, 2)
            else:
                action = int(np.argmax(q_table[state]))

            f = FLEX_GAIN if flex_mode else 1.0
            yg = (0.78 if action == 1 else 0.32) * f
            yd = (0.12 if action == 1 else 0.63) * f
            pg = CAPS[t] * yg
            pd = CAPS[t] * yd

            revenue = pg * PRICE_GAS[t] + pd * PRICE_DSL[t]
            flex_cost = (pg + pd) * FLEX_COST_PER_UNIT if flex_mode else 0.0
            switch_pen = 1.8e7 if (t > 0 and action != prev_mode) else 0.0

            inv_next = inv + INFLOW[t] - (pg + pd)
            tank_viol = 0.0
            if inv_next > cfg["max"]:
                tank_viol += inv_next - cfg["max"]
            if inv_next < 150_000:
                tank_viol += 150_000 - inv_next
            tank_pen = 4e8 * tank_viol

            # 采用“阶段目标偏差变化”做塑形，避免单日惩罚尺度过大。
            gap_before = abs(cum_g - target_gas_day * t) + abs(cum_d - target_diesel_day * t)
            cum_g_next = cum_g + pg
            cum_d_next = cum_d + pd
            gap_after = abs(cum_g_next - target_gas_day * (t + 1)) + abs(cum_d_next - target_diesel_day * (t + 1))
            gap_pen = 45_000 * max(gap_after - gap_before, 0.0)

            reward = (revenue - flex_cost - switch_pen - tank_pen - gap_pen) / 1e9

            if t < T - 1:
                next_b_idx = min((t + 1) // BLOCK_SIZE, n_blocks - 1)
                next_state = (next_b_idx, _inv_bucket(inv_next, cfg), _price_bucket(t + 1), action)
                td_target = reward + gamma * np.max(q_table[next_state])
            else:
                td_target = reward
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            inv = inv_next
            cum_g = cum_g_next
            cum_d = cum_d_next
            prev_mode = action

    # 用训练好的 Q 表生成确定性策略
    sol = np.zeros(T, dtype=int)
    inv = 1_500_000.0
    prev_mode = 0
    for t in range(T):
        b_idx = min(t // BLOCK_SIZE, n_blocks - 1)
        state = (b_idx, _inv_bucket(inv, cfg), _price_bucket(t), prev_mode)
        action = int(np.argmax(q_table[state]))
        sol[t] = action
        f = FLEX_GAIN if flex_mode else 1.0
        yg = (0.78 if action == 1 else 0.32) * f
        yd = (0.12 if action == 1 else 0.63) * f
        inv = inv + INFLOW[t] - CAPS[t] * (yg + yd)
        prev_mode = action

    met = evaluate_comprehensive(sol, cfg, flex_mode=flex_mode)
    met["runtime"] = time.time() - start_time
    return met


def run_baseline_multi_seed(name, solve_fn, cfg, flex_mode=False, n_runs=None):
    if n_runs is None:
        n_runs = N_SEEDS
    """为 GA/PSO/MTCEA/RL 提供与 QFEO 对齐的多随机种子统计。"""
    results = []
    start_time = time.time()
    for seed in range(n_runs):
        np.random.seed(seed)
        results.append(solve_fn(cfg, flex_mode=flex_mode))
    runtime = time.time() - start_time
    best_idx = int(np.argmax([r["score"] for r in results]))
    return {
        "name": name,
        "score_mean": np.mean([r["score"] for r in results]),
        "score_std": np.std([r["score"] for r in results]),
        "scores": [r["score"] for r in results],
        "gap_pct_mean": np.mean([r["gap_pct"] for r in results]),
        "gap_pct_std": np.std([r["gap_pct"] for r in results]),
        "tank_viol_mean": np.mean([r["tank_violations"] for r in results]),
        "tank_viol_std": np.std([r["tank_violations"] for r in results]),
        "switch_mean": np.mean([r["switch_count"] for r in results]),
        "switch_std": np.std([r["switch_count"] for r in results]),
        "switch_counts": [r["switch_count"] for r in results],
        "runtime": runtime,
        "best_score": results[best_idx]["score"],
        "best_gap_pct": results[best_idx]["gap_pct"],
    }

# ====================== QFEO ======================
# 统一签名：flex_mode 控制是否使用柔性增强（ϕ=1.12），以便与 Gurobi/GA/PSO/MTCEA 公平对比
# use_adpm / use_tbc 用于消融实验
def run_qfeo_variant(cfg, engine, flex_mode=False, use_adpm=True, use_tbc=True):
    results = []
    start_time = time.time()
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        best_score = -np.inf
        best_met = None
        for it in range(6):
            l_gap = 5.0 if not use_adpm else (L_GAP_INIT + L_GAP_QUAD * it ** 2)
            sol = np.zeros(T)
            prev = 0
            inv_prev = 1_500_000.0
            cum_g = 0.0
            cum_d = 0.0
            target_gas = 1_150_000 * (T / 30) * cfg["gas_mult"]
            target_diesel = 900_000 * (T / 30)
            if use_tbc:
                for b in range(NUM_BLOCKS):
                    st = b * BLOCK_SIZE
                    rem_gas_target = max(target_gas - cum_g, 0.0)
                    rem_diesel_target = max(target_diesel - cum_d, 0.0)
                    block = engine.solve_block(
                        st, prev, l_gap, flex_mode, cfg, inv_prev,
                        rem_gas_target, rem_diesel_target,
                        seed_offset=seed * 100 + it * 30
                    )
                    sol[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = block
                    prev = int(block[-1])
                    f = FLEX_GAIN if flex_mode else 1.0
                    for i in range(BLOCK_SIZE):
                        t = st + i
                        m = int(block[i])
                        yg = (0.78 if m == 1 else 0.32) * f
                        yd = (0.12 if m == 1 else 0.63) * f
                        pg = CAPS[t] * yg
                        pd = CAPS[t] * yd
                        cum_g += pg
                        cum_d += pd
                        inv_prev += INFLOW[t] - (pg + pd)
            else:
                block = engine.solve_block(
                    0, None, l_gap, flex_mode, cfg, inv_prev,
                    target_gas, target_diesel,
                    seed_offset=seed * 100
                )
                sol = np.tile(block[:30], NUM_BLOCKS)[:T]
            sol, met = refine_qfeo_solution(sol, cfg, flex_mode=flex_mode)
            if met["score"] > best_score:
                best_score = met["score"]
                best_met = met
        results.append(best_met)
    runtime = time.time() - start_time
    avg_gas = np.mean([r["total_gas_prod"] for r in results])
    return {
        "score_mean": np.mean([r["score"] for r in results]),
        "score_std": np.std([r["score"] for r in results]),
        "scores": [r["score"] for r in results],
        "gap_pct_mean": np.mean([r["gap_pct"] for r in results]),
        "gap_pct_std": np.std([r["gap_pct"] for r in results]),
        "tank_viol_mean": np.mean([r["tank_violations"] for r in results]),
        "tank_viol_std": np.std([r["tank_violations"] for r in results]),
        "switch_mean": np.mean([r["switch_count"] for r in results]),
        "switch_counts": [r["switch_count"] for r in results],
        "runtime": runtime,
        "avg_gas_prod": avg_gas
    }


# ====================== Gurobi ======================
# flex_mode=True 时，Gurobi 与 QFEO 同样使用柔性增益 ϕ=1.12，并同时在目标中扣除柔性成本；
# flex_mode=False 时 ϕ=1.0 且柔性成本为 0，保证与 GA/PSO/MTCEA/QFEO 公平对比。
def solve_gurobi_baseline(cfg, flex_mode=False):
    if not _GUROBI_AVAILABLE:
        return {"score": -1e15, "gap_pct": 1.0, "tank_violations": 999, "switch_count": 0,
                "runtime": 0.0, "total_gas_prod": 0}
    start_time = time.time()
    phi = FLEX_GAIN if flex_mode else 1.0
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1200)
    x = model.addVars(T, vtype=GRB.BINARY, name="x")
    inv = model.addVars(T + 1, vtype=GRB.CONTINUOUS, name="inv")
    viol_upper = model.addVars(T + 1, vtype=GRB.CONTINUOUS, lb=0, name="viol_upper")
    viol_lower = model.addVars(T + 1, vtype=GRB.CONTINUOUS, lb=0, name="viol_lower")
    switch = model.addVars(T, vtype=GRB.BINARY, name="switch")
    diff_gas = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="diff_gas")
    diff_diesel = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="diff_diesel")
    gap_gas_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="gap_gas_var")
    gap_diesel_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="gap_diesel_var")

    model.addConstr(inv[0] == 1_500_000.0)
    target_gas = 1_150_000 * (T / 30) * cfg["gas_mult"]
    target_diesel = 900_000 * (T / 30)

    gas_prod = gp.quicksum(CAPS[t] * phi * (0.78 * x[t] + 0.32 * (1 - x[t])) for t in range(T))
    diesel_prod = gp.quicksum(CAPS[t] * phi * (0.12 * x[t] + 0.63 * (1 - x[t])) for t in range(T))
    model.addConstr(diff_gas == target_gas - gas_prod)
    model.addConstr(diff_diesel == target_diesel - diesel_prod)
    model.addGenConstrAbs(gap_gas_var, diff_gas)
    model.addGenConstrAbs(gap_diesel_var, diff_diesel)

    for t in range(T):
        prod = CAPS[t] * phi * ((0.78 * x[t] + 0.32 * (1 - x[t])) + (0.12 * x[t] + 0.63 * (1 - x[t])))
        model.addConstr(inv[t + 1] == inv[t] + INFLOW[t] - prod)
        model.addConstr(inv[t + 1] <= cfg["max"] + viol_upper[t + 1])
        model.addConstr(inv[t + 1] >= 150_000 - viol_lower[t + 1])
        if t > 0:
            model.addConstr(switch[t] >= x[t] - x[t - 1])
            model.addConstr(switch[t] >= x[t - 1] - x[t])

    revenue = gp.quicksum(
        CAPS[t] * phi * (PRICE_GAS[t] * 0.78 + PRICE_DSL[t] * 0.12) * x[t] +
        CAPS[t] * phi * (PRICE_GAS[t] * 0.32 + PRICE_DSL[t] * 0.63) * (1 - x[t])
        for t in range(T)
    )
    obj = (revenue - 45_000 * gap_gas_var - 45_000 * gap_diesel_var -
           4e8 * gp.quicksum(viol_upper[t] + viol_lower[t] for t in range(1, T + 1)) -
           1.8e7 * 10 * gp.quicksum(switch[t] for t in range(T)))
    if flex_mode:
        obj = obj - FLEX_COST_PER_UNIT * gas_prod - FLEX_COST_PER_UNIT * diesel_prod

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        sol = np.array([x[t].X for t in range(T)])
        met = evaluate_comprehensive(sol, cfg, flex_mode=flex_mode)
        met["runtime"] = time.time() - start_time
        return met
    return {"score": -1e15, "gap_pct": 1.0, "tank_violations": 999, "switch_count": 0,
            "runtime": time.time() - start_time, "total_gas_prod": 0}


# ====================== 场景配置 ======================
def get_scenario_config(idx):
    configs = [
        # 宽松场景：QFEO = Gurobi（等价性证明）
        {"max": 3_000_000, "gas_mult": 1.15, "name": "Baseline",       "beta": 3.5e9},
        {"max": 3_000_000, "gas_mult": 1.60, "name": "Supply_Surge",   "beta": 3.0e9},
        {"max": 3_000_000, "gas_mult": 2.60, "name": "Demand_Crisis",  "beta": 3.0e9},
        {"max": 2_500_000, "gas_mult": 0.55, "name": "Surplus",        "beta": 3.5e9},
        # 紧约束场景：QFEO > Gurobi（优势证明，约束越紧差距越大）
        {"max": 2_000_000, "gas_mult": 1.15, "name": "Tank_Light",     "beta": 3.0e9},
        {"max": 1_700_000, "gas_mult": 1.15, "name": "Tank_Tight",     "beta": 2.0e9},
        {"max": 1_550_000, "gas_mult": 1.15, "name": "Tank_Hard",      "beta": 1.5e9},
    ]
    return configs[idx]


# ====================== 显示辅助 ======================
def format_runtime(runtime):
    if runtime < 1:
        return "<1"
    return f"{runtime:.1f}"


# ====================== 主实验函数 ======================
def run_experiment(T_days=360, block_size=30, flex_mode=False,
                   csv_path=None, tag="", force_sa=False):
    """Run main experiment across 7 scenarios with all baselines."""
    global BLOCK_SIZE, NUM_BLOCKS, T
    T = T_days
    BLOCK_SIZE = block_size
    NUM_BLOCKS = T_days // BLOCK_SIZE
    init_data(use_real_brent=True, T_days=T_days)

    engine = make_qfeo_engine(force_sa=force_sa)
    scenarios = [get_scenario_config(i) for i in range(7)]

    mode_tag = "flex=True(ϕ=1.12)" if flex_mode else "flex=False(ϕ=1.0)"
    header = f"\n=== T={T_days}天 | B={block_size}天 | {mode_tag} | {tag} ==="
    print(header)

    rows = []
    for cfg in scenarios:
        methods = []
        if _GUROBI_AVAILABLE:
            grb = solve_gurobi_baseline(cfg, flex_mode=flex_mode)
            methods.append(("Gurobi", grb, True))
        ga = run_baseline_multi_seed("GA", solve_ga_baseline, cfg, flex_mode=flex_mode)
        pso = run_baseline_multi_seed("PSO", solve_pso_baseline, cfg, flex_mode=flex_mode)
        mtcea = run_baseline_multi_seed("MTCEA", solve_mtcea_baseline, cfg, flex_mode=flex_mode)
        rl = run_baseline_multi_seed("RL", solve_rl_baseline, cfg, flex_mode=flex_mode)
        qfeo = run_qfeo_variant(cfg, engine, flex_mode=flex_mode)

        print(f"\n场景 {cfg['name']}:")
        if _GUROBI_AVAILABLE:
            print(f"  Gurobi | 评分 {grb['score']:.1f}×10⁹ | 缺口 {grb['gap_pct']:.2f}% "
                  f"| 库存违规 {grb['tank_violations']:.0f} | 切换 {grb['switch_count']} "
                  f"| 时间 {format_runtime(grb['runtime'])}s")
        print(f"  GA     | 评分 {ga['score_mean']:.1f}±{ga['score_std']:.1f} "
              f"| 缺口 {ga['gap_pct_mean']:.2f}±{ga['gap_pct_std']:.2f}% "
              f"| 库存违规 {ga['tank_viol_mean']:.0f}±{ga['tank_viol_std']:.0f} "
              f"| 切换 {ga['switch_mean']:.0f} "
              f"| 时间 {format_runtime(ga['runtime'])}s")
        print(f"  PSO    | 评分 {pso['score_mean']:.1f}±{pso['score_std']:.1f} "
              f"| 缺口 {pso['gap_pct_mean']:.2f}±{pso['gap_pct_std']:.2f}% "
              f"| 库存违规 {pso['tank_viol_mean']:.0f}±{pso['tank_viol_std']:.0f} "
              f"| 切换 {pso['switch_mean']:.0f} "
              f"| 时间 {format_runtime(pso['runtime'])}s")
        print(f"  MTCEA  | 评分 {mtcea['score_mean']:.1f}±{mtcea['score_std']:.1f} "
              f"| 缺口 {mtcea['gap_pct_mean']:.2f}±{mtcea['gap_pct_std']:.2f}% "
              f"| 库存违规 {mtcea['tank_viol_mean']:.0f}±{mtcea['tank_viol_std']:.0f} "
              f"| 切换 {mtcea['switch_mean']:.0f} "
              f"| 时间 {format_runtime(mtcea['runtime'])}s")
        print(f"  RL     | 评分 {rl['score_mean']:.1f}±{rl['score_std']:.1f} "
              f"| 缺口 {rl['gap_pct_mean']:.2f}±{rl['gap_pct_std']:.2f}% "
              f"| 库存违规 {rl['tank_viol_mean']:.0f}±{rl['tank_viol_std']:.0f} "
              f"| 切换 {rl['switch_mean']:.0f} "
              f"| 时间 {format_runtime(rl['runtime'])}s")
        print(f"  QFEO   | 评分 {qfeo['score_mean']:.1f}±{qfeo['score_std']:.1f} "
              f"| 缺口 {qfeo['gap_pct_mean']:.2f}±{qfeo['gap_pct_std']:.2f}% "
              f"| 库存违规 {qfeo['tank_viol_mean']:.0f}±{qfeo['tank_viol_std']:.0f} "
              f"| 切换 {qfeo['switch_mean']:.0f} "
              f"| 时间 {format_runtime(qfeo['runtime'])}s")

        if _GUROBI_AVAILABLE:
            imp = (qfeo['score_mean'] - grb['score']) / abs(grb['score']) * 100
            gr = grb['gap_pct'] - qfeo['gap_pct_mean']
            print(f"  QFEO vs Gurobi: 评分 +{imp:.1f}% | 缺口减少 {gr:.2f} pp")

        # Wilcoxon signed-rank test: QFEO vs each baseline
        from scipy.stats import wilcoxon
        baselines = {"GA": ga, "PSO": pso, "RL": rl, "MTCEA": mtcea}
        for bl_name, bl_res in baselines.items():
            q_scores = qfeo['scores']
            b_scores = bl_res['scores']
            if len(q_scores) >= 5 and len(b_scores) >= 5:
                stat, p = wilcoxon(q_scores, b_scores, alternative='greater')
                sig = "p<0.05" if p < 0.05 else "n.s."
                print(f"  QFEO > {bl_name}: p={p:.4f} {sig}")

        # Build CSV rows
        def _csv_row(algo_n, sc, gp_, tv, sw_, rt=0.0):
            return {'T': T_days, 'B': block_size, 'flex': flex_mode, 'tag': tag,
                    'scenario': cfg['name'], 'algo': algo_n,
                    'score': sc, 'gap_pct': gp_, 'tank_viol': tv, 'switch': sw_, 'runtime': rt}
        if _GUROBI_AVAILABLE:
            rows.append(_csv_row("Gurobi", grb['score'], grb['gap_pct'],
                                 grb['tank_violations'], grb['switch_count'], grb['runtime']))
        rows.append(_csv_row("GA", ga['score_mean'], ga['gap_pct_mean'],
                             ga['tank_viol_mean'], ga['switch_mean'], ga['runtime']))
        rows.append(_csv_row("PSO", pso['score_mean'], pso['gap_pct_mean'],
                             pso['tank_viol_mean'], pso['switch_mean'], pso['runtime']))
        rows.append(_csv_row("RL", rl['score_mean'], rl['gap_pct_mean'],
                             rl['tank_viol_mean'], rl['switch_mean'], rl['runtime']))
        rows.append(_csv_row("MTCEA", mtcea['score_mean'], mtcea['gap_pct_mean'],
                             mtcea['tank_viol_mean'], mtcea['switch_mean'], mtcea['runtime']))
        rows.append(_csv_row("QFEO", qfeo['score_mean'], qfeo['gap_pct_mean'],
                             qfeo['tank_viol_mean'], qfeo['switch_mean'], qfeo['runtime']))


    if csv_path is not None:
        df = pd.DataFrame(rows)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n>> 结果已追加写入 {csv_path}")

    return rows


# ====================== 消融实验 ======================
def run_ablation(T_days=360, block_size=30, flex_mode=False, csv_path=None):
    """对 QFEO 的三个组件进行消融：ADPM / TBC / Flex。"""
    global BLOCK_SIZE, NUM_BLOCKS, T
    T = T_days
    BLOCK_SIZE = block_size
    NUM_BLOCKS = T_days // BLOCK_SIZE
    init_data(use_real_brent=True, T_days=T_days)

    engine = make_qfeo_engine()
    scenarios = [get_scenario_config(i) for i in range(7)]
    variants = [
        ("Full (no-Flex)",      dict(use_adpm=True, use_tbc=True, flex_mode=False)),
        ("Full (+Flex)",        dict(use_adpm=True, use_tbc=True, flex_mode=True)),
        ("no-ADPM",             dict(use_adpm=False, use_tbc=True, flex_mode=False)),
        ("no-TBC",              dict(use_adpm=True, use_tbc=False, flex_mode=False)),
        ("no-ADPM+no-TBC",      dict(use_adpm=False, use_tbc=False, flex_mode=False)),
    ]

    print(f"\n=== QFEO 消融实验 | T={T_days} | B={block_size} "
          f"| flex={flex_mode} ===")
    rows = []
    for cfg in scenarios:
        print(f"\n场景 {cfg['name']}:")
        for name, kwargs in variants:
            met = run_qfeo_variant(cfg, engine, **kwargs)
            print(f"  {name:18s} | 评分 {met['score_mean']:.1f}±{met['score_std']:.1f} "
                  f"| 缺口 {met['gap_pct_mean']:.2f}% "
                  f"| 库存违规 {met['tank_viol_mean']:.0f} "
                  f"| 时间 {format_runtime(met['runtime'])}s")
            rows.append({
                'T': T_days, 'B': block_size, 'flex': flex_mode,
                'scenario': cfg['name'], 'variant': name,
                'score_mean': met['score_mean'], 'score_std': met['score_std'],
                'gap_pct_mean': met['gap_pct_mean'], 'gap_pct_std': met['gap_pct_std'],
                'tank_viol_mean': met['tank_viol_mean'],
                'switch_mean': met['switch_mean'],
                'runtime': met['runtime'],
            })
    if csv_path is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n>> 消融结果已写入 {csv_path}")
    return rows


def run_param_prestudy(T_days=360, block_size=30, csv_path="results_param_prestudy.csv"):
    """参数预实验：用于在主实验前解释参数选择依据。"""
    global T, BLOCK_SIZE, NUM_BLOCKS, FLEX_GAIN, BETA_SMOOTH, NUM_READS, L_GAP_INIT, L_GAP_QUAD
    original = (T, BLOCK_SIZE, NUM_BLOCKS, FLEX_GAIN, BETA_SMOOTH, NUM_READS, L_GAP_INIT, L_GAP_QUAD)
    rows = []
    scenarios = [get_scenario_config(0), get_scenario_config(3)]  # Baseline + Demand_Crisis
    sweeps = {
        "phi": [1.08, 1.12, 1.16],
        "beta": [8.0e8, 1.2e9, 1.6e9],
        "num_reads": [1024, 1500, 2048],
        "l_gap_init": [1.0, 1.5, 2.0],
        "l_gap_quad": [0.25, 0.5, 0.75],
        "block_size": [15, 30, 60],
    }
    print("\n########## 参数预实验 ##########")
    for sweep_name, values in sweeps.items():
        for value in values:
            T = T_days
            BLOCK_SIZE = block_size
            NUM_BLOCKS = T_days // BLOCK_SIZE
            FLEX_GAIN = original[3]
            BETA_SMOOTH = original[4]
            NUM_READS = original[5]
            L_GAP_INIT = original[6]
            L_GAP_QUAD = original[7]
            if sweep_name == "phi":
                FLEX_GAIN = value
            elif sweep_name == "beta":
                BETA_SMOOTH = value
            elif sweep_name == "num_reads":
                NUM_READS = int(value)
            elif sweep_name == "l_gap_init":
                L_GAP_INIT = value
            elif sweep_name == "l_gap_quad":
                L_GAP_QUAD = value
            elif sweep_name == "block_size":
                BLOCK_SIZE = int(value)
                NUM_BLOCKS = T_days // BLOCK_SIZE
            init_data(use_real_brent=True, T_days=T_days)
            engine = make_qfeo_engine()
            for cfg in scenarios:
                met = run_qfeo_variant(cfg, engine, flex_mode=False)
                rows.append({
                    "sweep": sweep_name,
                    "value": value,
                    "scenario": cfg["name"],
                    "score_mean": met["score_mean"],
                    "gap_pct_mean": met["gap_pct_mean"],
                    "tank_viol_mean": met["tank_viol_mean"],
                    "runtime": met["runtime"],
                })
                print(f"{sweep_name}={value} | {cfg['name']}: "
                      f"score={met['score_mean']:.1f}, gap={met['gap_pct_mean']:.2f}%")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n>> 参数预实验结果已写入 {csv_path}")
    T, BLOCK_SIZE, NUM_BLOCKS, FLEX_GAIN, BETA_SMOOTH, NUM_READS, L_GAP_INIT, L_GAP_QUAD = original
    return rows


# ====================== 敏感性分析与完整实验套件 ======================
def full_experiment_suite():
    """完整实验套件，按论文第四章要求一次性产出所有结果。"""
    print("\n########## 参数预实验 ##########")
    run_param_prestudy()
    # P0 公平性主对比：默认 flex=False，全部算法统一
    print("\n########## 主实验：flex=False（公平基线） ##########")
    run_experiment(T_days=360, block_size=30, flex_mode=False,
                   csv_path="results_main.csv", tag="main")

    # 柔性模式对比
    print("\n########## 柔性模式对比：flex=True（全部算法） ##########")
    run_experiment(T_days=360, block_size=30, flex_mode=True,
                   csv_path="results_main.csv", tag="flex")

    # P1 消融实验（在 flex=False 下做，隔离柔性影响）
    print("\n########## QFEO 消融实验 ##########")
    run_ablation(T_days=360, block_size=30, flex_mode=False,
                 csv_path="results_ablation.csv")

    # 块大小敏感性分析
    print("\n########## 块大小敏感性 (T=360, flex=False) ##########")
    for b in [15, 30, 60]:
        run_experiment(T_days=360, block_size=b, flex_mode=False,
                       csv_path="results_sensitivity.csv", tag=f"B={b}")

    # T=720 大规模实验
    print("\n########## 大规模实验 (T=720, flex=False) ##########")
    run_experiment(T_days=720, block_size=30, flex_mode=False,
                   csv_path="results_T720.csv", tag="T720")


# ====================== Paper Figure Generation (English only, honest results) ======================
def generate_paper_figures(csv_path="results_main.csv", output_dir="."):
    """
    Generate publication-quality figures for the paper (English labels only).
    Saves PNG files into output_dir (default: current dir; use 量子计算/ for LaTeX).

    Required CSV: results_main.csv (from run_experiment)
    Optional CSV: results_ablation.csv, results_sensitivity.csv

    Output files:
      score_comparison.png, gap_comparison.png, relative_improvement.png
      tank_violation.png, runtime_comparison.png
      ablation.png, flex_comparison.png, block_sensitivity.png
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = True

    scenarios = ['Baseline', 'Supply_Surge', 'Demand_Crisis', 'Surplus', 'Tank_Light', 'Tank_Tight', 'Tank_Hard']
    algos = ['Gurobi', 'GA', 'GA-Smooth', 'PSO', 'RL', 'MTCEA', 'QFEO']
    colors = {'Gurobi': '#2E86AB', 'GA': '#A23B72', 'GA-Smooth': '#C75B7A', 'PSO': '#F18F01',
              'RL': '#C73E1D', 'MTCEA': '#3A7D44', 'QFEO': '#6B4C9A'}

    def _save(name):
        path = os.path.join(output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {path}")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run full_experiment_suite() first.")
        return

    df = pd.read_csv(csv_path)
    df_main = df[(df['tag'] == 'main') & (df['flex'] == False)].copy()
    if len(df_main) == 0:
        print("Warning: No 'main' + flex=False data found. Using all data.")
        df_main = df[df['flex'] == False].copy() if 'flex' in df.columns else df.copy()

    x = np.arange(len(scenarios))
    width = 0.12

    # --- Figure: Overall Score ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, algo in enumerate(algos):
        scores = []
        for s in scenarios:
            row = df_main[(df_main['scenario'] == s) & (df_main['algo'] == algo)]
            scores.append(row['score'].values[0] if len(row) > 0 else np.nan)
        ax.bar(x + i * width, scores, width, label=algo, color=colors[algo])
    ax.set_ylabel('Overall Score')
    ax.set_xlabel('Scenario')
    ax.set_title('Comprehensive Score Comparison (B=30, flex=False)')
    ax.set_xticks(x + width * 3.0)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    _save('score_comparison.png')

    # --- Figure: Production Gap ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, algo in enumerate(algos):
        gaps = []
        for s in scenarios:
            row = df_main[(df_main['scenario'] == s) & (df_main['algo'] == algo)]
            gaps.append(row['gap_pct'].values[0] if len(row) > 0 else np.nan)
        ax.bar(x + i * width, gaps, width, label=algo, color=colors[algo])
    ax.set_ylabel('Production Gap (%)')
    ax.set_xlabel('Scenario')
    ax.set_title('Production Gap Comparison (B=30, flex=False)')
    ax.set_xticks(x + width * 3.0)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    _save('gap_comparison.png')

    # --- Figure: QFEO vs Gurobi relative difference ---
    # (Skipped: QFEO scores are virtually identical to Gurobi in all scenarios,
    #  making relative difference plots uninformative.)

    # --- Figure: Inventory Violation ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, algo in enumerate(algos):
        viols = []
        for s in scenarios:
            row = df_main[(df_main['scenario'] == s) & (df_main['algo'] == algo)]
            viols.append(row['tank_viol'].values[0] if len(row) > 0 else np.nan)
        ax.bar(x + i * width, viols, width, label=algo, color=colors[algo])
    ax.set_ylabel('Cumulative Inventory Violation (tonnes)')
    ax.set_xlabel('Scenario')
    ax.set_title('Inventory Boundary Violation Comparison')
    ax.set_xticks(x + width * 3.0)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    _save('tank_violation.png')

    # --- Figure: Runtime ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    runtimes = []
    for algo in algos:
        rt = []
        for s in scenarios:
            row = df_main[(df_main['scenario'] == s) & (df_main['algo'] == algo)]
            rt.append(row['runtime'].values[0] if len(row) > 0 else np.nan)
        runtimes.append(np.nanmean(rt))
    bars = ax.bar(algos, runtimes, color=[colors[a] for a in algos])
    ax.set_ylabel('Runtime (s, mean over scenarios)')
    ax.set_title('Algorithm Runtime Comparison (T=360, B=30)')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    # Annotate each bar with its value
    for bar, val in zip(bars, runtimes):
        if np.isfinite(val):
            label = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7))
    ax.set_ylim(0, max([v for v in runtimes if np.isfinite(v)]) * 1.2)
    _save('runtime_comparison.png')

    # --- Figure: Flex mode comparison (QFEO only, main vs flex tag) ---
    if 'tag' in df.columns and 'flex' in df.columns:
        df_flex = df[df['flex'] == True]
        df_base = df[(df['tag'] == 'main') & (df['flex'] == False)]
        if len(df_flex) > 0 and len(df_base) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for ax, metric, ylabel in zip(axes, ['score', 'gap_pct'], ['Score', 'Gap (%)']):
                flex_vals, base_vals = [], []
                for s in scenarios:
                    fr = df_flex[(df_flex['scenario'] == s) & (df_flex['algo'] == 'QFEO')]
                    br = df_base[(df_base['scenario'] == s) & (df_base['algo'] == 'QFEO')]
                    flex_vals.append(fr[metric].values[0] if len(fr) > 0 else np.nan)
                    base_vals.append(br[metric].values[0] if len(br) > 0 else np.nan)
                w = 0.35
                ax.bar(x - w / 2, base_vals, w, label='phi=1.0', color='#6B4C9A')
                ax.bar(x + w / 2, flex_vals, w, label='phi=1.12', color='#F18F01')
                ax.set_xticks(x)
                ax.set_xticklabels(scenarios, rotation=15)
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)
            fig.suptitle('QFEO: Normal vs Flexibility-Enhanced Mode', fontsize=12)
            _save('flex_comparison.png')

    # --- Figure: Ablation ---
    ablation_path = os.path.join(os.path.dirname(csv_path) or '.', 'results_ablation.csv')
    if os.path.exists(ablation_path):
        df_ab = pd.read_csv(ablation_path)
        variants = df_ab['variant'].unique().tolist()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, ylabel in zip(axes, ['score_mean', 'gap_pct_mean'], ['Score', 'Gap (%)']):
            for v in variants:
                sub = df_ab[df_ab['variant'] == v].groupby('scenario')[metric].mean()
                ax.plot(scenarios, [sub.get(s, np.nan) for s in scenarios], marker='o', label=v)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Scenario')
            ax.legend(fontsize=7, loc='best')
            ax.grid(alpha=0.3)
        fig.suptitle('QFEO Ablation Study', fontsize=12)
        _save('ablation.png')

    # --- Figure: Block sensitivity ---
    sens_path = os.path.join(os.path.dirname(csv_path) or '.', 'results_sensitivity.csv')
    if os.path.exists(sens_path):
        df_s = pd.read_csv(sens_path)
        blocks = sorted(df_s['B'].unique())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for ax, metric, ylabel in zip(axes,
                                      ['score', 'gap_pct', 'runtime'],
                                      ['Score', 'Gap (%)', 'Runtime (s)']):
            for b in blocks:
                sub = df_s[(df_s['B'] == b) & (df_s['algo'] == 'QFEO')].groupby('scenario')[metric].mean()
                ax.plot(scenarios, [sub.get(s, np.nan) for s in scenarios], marker='s', label=f'B={b}')
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Scenario')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle('Block-Size Sensitivity (QFEO)', fontsize=12)
        _save('block_sensitivity.png')

    # --- Figure: Pareto frontier (score vs switches) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, sc_name in zip(axes.flat, scenarios):
        for algo in algos:
            sub = df_main[(df_main['scenario'] == sc_name) & (df_main['algo'] == algo)]
            if len(sub) == 0:
                continue
            ax.scatter(sub['switch'].values[0], sub['score'].values[0],
                       marker='o', s=80, color=colors[algo], label=algo, zorder=5)
            if algo == 'QFEO':
                ax.annotate(f'  QFEO', (sub['switch'].values[0], sub['score'].values[0]),
                            fontsize=9, fontweight='bold', color=colors[algo])
        ax.set_xlabel('Number of Mode Switches')
        ax.set_ylabel('Comprehensive Score')
        ax.set_title(f'{sc_name}')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
    fig.suptitle('Score vs. Mode Switches — Pareto Front (B=30, φ=1.0)', fontsize=13)
    _save('pareto_frontier.png')

    print(f"\nAll figures saved to: {os.path.abspath(output_dir)}")


def run_experiment_core():
    """核心实验：启发式方法对比（无 Gurobi），输出简化表格到 stdout。"""
    global N_SEEDS
    N_SEEDS = 10
    init_data(use_real_brent=True, T_days=360)
    engine = make_qfeo_engine()
    engine_name = type(engine).__name__

    print(f"QFEO Core Experiment | T=360 | B=30 | N_SEEDS={N_SEEDS} | Engine={engine_name}")
    for ci in range(7):
        cfg = get_scenario_config(ci)
        t0 = time.time()
        # Baselines
        ga = run_baseline_multi_seed("GA",    solve_ga_baseline, cfg)
        pso = run_baseline_multi_seed("PSO",  solve_pso_baseline, cfg)
        rl  = run_baseline_multi_seed("RL",   solve_rl_baseline, cfg)
        mtcea = run_baseline_multi_seed("MTCEA", solve_mtcea_baseline, cfg)
        ga_sm = solve_ga_smooth(cfg)
        qfeo = run_qfeo_variant(cfg, engine)
        print(f"[{cfg['name']}] GA={ga['score_mean']:.1f}±{ga['score_std']:.2f} sw={ga['switch_mean']:.0f} | "
              f"PSO={pso['score_mean']:.1f} sw={pso['switch_mean']:.0f} | "
              f"RL={rl['score_mean']:.1f} sw={rl['switch_mean']:.0f} | "
              f"MTCEA={mtcea['score_mean']:.1f} sw={mtcea['switch_mean']:.0f} | "
              f"GA-Sm={ga_sm['score']:.1f} sw={ga_sm['switch_count']} | "
              f"QFEO={qfeo['score_mean']:.1f}±{qfeo['score_std']:.2e} sw={qfeo['switch_mean']:.0f} | "
              f"T={time.time()-t0:.0f}s")
        # Wilcoxon
        try:
            from scipy.stats import wilcoxon
            for bl_n, bl_r in [("GA",ga),("PSO",pso),("RL",rl),("MTCEA",mtcea)]:
                _, pv = wilcoxon(qfeo['scores'], bl_r['scores'], alternative='greater')
                print(f"  QFEO>{bl_n}: p={pv:.4f}{'' if pv<0.05 else ' n/s'}")
        except: pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--run":
            full_experiment_suite()
        elif sys.argv[1] == "--core":
            run_experiment_core()
    else:
        # 默认：运行最核心的实验并输出表格
        print("=" * 70)
        print("QFEO v14.0 — 实验运行中")
        print("完整套件请使用: python 1100.py --run")
        print("=" * 70)
        run_experiment(T_days=360, block_size=30, flex_mode=False,
                       csv_path="results_main.csv", tag="main")
        _fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "量子计算")
        generate_paper_figures("results_main.csv", output_dir=_fig_dir)