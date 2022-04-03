# Neural Decision Tree Method Development in Matlab

### The general sequence to run the NDT and PINE algorithm comparison: 
1. `startup.m`: setup environment
2. `featureInGeneral.m`: create NDT and PINE model
3. `evalNdt.m`: Figure comparison of NDT vs PINE

### Details of key files:
- `startup.m` file: change data dir to your own with folder `results`, `PINE_data`, and `figures`.
    - `results`: store trained NDT model
    - `PINE_data`: store raw data set
    - `figures`: store matlab figures

- `featureXxx.m` file: Train NDT algorithm based on different feature scaling:
    - `featureInGeneral.m`: Run NDT and PINE with no additional feature scaling
    - `featurePolarMLT.m`: Run NDT and PINE with converting `MLT` into sine and cosine cartesian coordinate.

- `evalXxx.m` file: evaluation of NDT and PINE
    - `evalNdt.m`: Evaluate NDT and PINE trained without scaling features (i.e. no statistic scale, no weight scaled) in RMSE and density plot
    - `evalPolarMLT.m`: Evaluate NDT and PINE trained with converting `MLT` into sine and cosine cartesian coordinate.

