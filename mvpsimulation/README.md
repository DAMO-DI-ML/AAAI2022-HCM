## Requirements
1. Download tetrad-gui-laucher from [project](https://cloud.ccd.pitt.edu/nexus/content/repositories/releases/edu/cmu/tetrad-gui)

## Simulation Steps
1. Set configuration
    - numMeasures: number of variables
    - avgDegree: average degree
    - percentDiscrete: percentage of discrete variables
    - meanLow/High: min/max value of intercept
    - coefLow/High: min/max value of linear coefficients
    - varLow/High: min/max value of noise variance
    - betaLow/High: min/max value of nonlinear coefficients (sine)
    
2. Make java new class

```
javac -cp "tetrad-gui-6.9.0-launch.jar" edu/cmu/tetrad/algcomparison/simulation/ConditionalGaussianSimulation.java

jar uf tetrad-gui-6.9.0-launch.jar edu/cmu/tetrad/algcomparison/simulation/ConditionalGaussianSimulation.class
jar uf tetrad-gui-6.9.0-launch.jar edu/cmu/tetrad/algcomparison/simulation/ConditionalGaussianSimulation$VariableValues.class
jar uf tetrad-gui-6.9.0-launch.jar edu/cmu/tetrad/algcomparison/simulation/ConditionalGaussianSimulation$Combination.class


javac -cp "tetrad-gui-6.9.0-launch.jar" edu/cmu/tetrad/algcomparison/examples/SaveMVPSimulations.java
jar uf tetrad-gui-6.9.0-launch.jar edu/cmu/tetrad/algcomparison/examples/SaveMVPSimulations.class 
java -cp tetrad-gui-6.9.0-launch.jar edu.cmu.tetrad.algcomparison.examples.SaveMVPSimulations

```
 
