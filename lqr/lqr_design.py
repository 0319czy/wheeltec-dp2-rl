"""二阶倒立摆（B585 DP2）离散 LQR 设计 —— 复现厂商 MATLAB 脚本并求解增益。

模型来源：附送资料 `MATLAB程序/R2022a/inverted_pendulum_on_self_balancing_robot.m`
状态量 x = [θL, θR, θ1, θ2, θL̇, θṘ, θ1̇, θ2̇]
  θL,θR  左右轮转角(rad)   θ1 车身倾角(rad)   θ2 摆杆角(rad)
输入 u = [T_L, T_R]（左右轮力矩，折算后为无量纲）
物理参数 m1=0.9kg(车身) m2=0.1kg(摆杆) r=0.0335m L1=0.126m L2=0.390m g=9.8
"""
import numpy as np
import scipy.linalg as sla

np.set_printoptions(precision=4, suppress=True, linewidth=140)

m1, m2, r, L1, L2, g = 0.9, 0.1, 0.0335, 0.126, 0.390, 9.8
l1, l2 = L1 / 2, L2 / 2
I1, I2 = (1/12) * m1 * L1**2, (1/12) * m2 * L2**2

# ---- 质量矩阵 p ----
p = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [r/2*(m1*l1 + m2*L1), r/2*(m1*l1 + m2*L1), m1*l1**2 + m2*L1**2 + I1, m2*L1*l2],
    [r/2*m2*l2,          r/2*m2*l2,          m2*L1*l2,                    m2*l2**2 + I2],
])
# ---- 重力与输入项 q ----
q = np.zeros((4, 10))
q[0, 8] = 1
q[1, 9] = 1
q[2, 2] = (m1*l1 + m2*L1) * g
q[3, 3] = m2 * g * l2

temp = np.linalg.inv(p) @ q
A = np.zeros((8, 8)); A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = 1
A[4:, :] = temp[:, :8]
B = np.zeros((8, 2)); B[4:, :] = temp[:, 8:10]

# ---- 离散化（Ts=10ms，零阶保持）----
Ts = 0.01
E = sla.expm(np.block([[A, B], [np.zeros((2, 10))]]) * Ts)
G, Hd = E[:8, :8], E[:8, 8:]

# ---- LQR 权重（厂商原值）----
Q = np.diag([51.2938, 51.2938, 32.8281, 131.3123, 51.2938, 51.2938, 131.3123, 131.3123])
R = 0.0005 * np.eye(2)


def solve(qw=None, rho=0.0005):
    Qm = np.diag(qw) if qw is not None else Q
    Rm = rho * np.eye(2)
    P = sla.solve_discrete_are(G, Hd, Qm, Rm)
    return np.linalg.inv(Rm + Hd.T @ P @ Hd) @ (Hd.T @ P @ G)


if __name__ == '__main__':
    print('=== 状态定义 ===')
    print('x = [θL, θR, θ1, θ2, θL̇, θṘ, θ1̇, θ2̇]^T')
    print('\n=== 连续 A (8x8) ===');  print(A)
    print('\n=== 连续 B (8x2) ===');  print(B)
    print(f'\n=== 离散 G (Ts={Ts}) 前 5 列 ===');  print(G[:, :5])
    print('\n=== Q diag ===');  print(np.diag(Q))
    print('=== R ===');         print(R)
    K = solve()
    print('\n=== LQR 增益 K (2x8) ===');  print(K)
    print('\n固件中的 K1 = [81.2695, -10.0616, -5492.4061, 18921.7098, 100.3633, 8.0376, 447.3084, 2962.7738]')
    print('固件中的 K2 = [-10.0616, 81.2695, -5492.4061, 18921.7098, 8.0376, 100.3633, 447.3084, 2962.7738]')
    print('\n复现误差 max|ΔK| = %.2e' % float(np.abs(K[0] - np.array([81.2695, -10.0616, -5492.4061, 18921.7098, 100.3633, 8.0376, 447.3084, 2962.7738])).max()))
