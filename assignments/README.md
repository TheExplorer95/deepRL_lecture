# deepRl assignments

**1. Assignment** (tabular Q-leraning)
- ipython notebook; tabular solution without ray; results inside notebook

**2. Assignment** (Q-learning with funct. approx.)
- solved with and without ray
- results only for ray solution inside results folder (loss, avg_steps)

**3. Assignment** (ActorCritic algorithm; polGradient update scaled by advantage)
- started off with wrong environment (MountainCar); wasn't able to solve, problems with exploration
- LunarLander solved? space ship learns to fly stable, but besides landing attempts it sometimes just hovers in the air (example assignment_3_LunarLandar/results/20210323-121215-polAdvantage-MaxStep250-ExplRange0.05-BatchSize128_Delay7_lr1e-07_L2reg)
- results: cumulative reward, avg_steps, actor loss, critic loss, model of the trained network

**Unfortunately** I wasn't able to make a fancy preview with the trained networks (started though), but plots should be sufficient, if not please contact me and I'll finish it up.

To **run the code**, install env from the respective Assignment (should be more or less the same) and run: 'python main.py' from the assignent folder


