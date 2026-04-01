This folder contains a fresh B-H curve analysis using the complex harmonic-fit model.

Model:
V(t) = a0 + sum[a_n sin(nwt) + b_n cos(nwt)], up to the harmonic order set in config.json.

Outputs are generated only from raw TEK traces in raw\ and written to:
- final\
- plots\

Key formulas used:
- H = N*Vx/(R*L)
- B = 0.5*Vy
- A_loop = 1/2 * |sum(H_i*B_{i+1} - B_i*H_{i+1})|
- E_loss = A_loop
- P_loss = f * E_loss
