# A. Custom-ML With Geometric Reasoning  capabilities
This Branch Contained Explanation regarding CronicalSpark AI, Also I Have Provided A TestBed for Cronical Spark itself that anyone can use and see the Performance of CronicalSpark 
Note = (Tho its a bit Buggy, The plot doesnt show up, but you can read the Performance metrics of the prints)
Hope you understand the math and its Functions. Have Fun Checking and testing. :)
Creator = Anonimity X.11 / Indonesia 

# CronicalSpark AI:
CronicalSpark is a Custom Complex Reasoning AI That Was Using Certainty as a Measurement of its Performance on task, This AI Performs A Good Certainty (without being overconfident swinging up down like Regular AI on little trainings)  In Linear And Sine Pattern Recognition, While Also Being Strategically Cautious and Opportunist At Random and Step Pattern. This AI can also Have Dual Mode, The Human in the Loop Intervention And Autonomous Mode, There is a dedicated Function for CronicalSpark Specifically For It if CronicalSpark needed Human judgement for feedback.

# Mathematical Foundation And Expression

The Mathematical Principle used to built FolderNet, Epsitron Transformer and Epsilon Policy used the Nonlinear Numpy syntax Such as:

```math
~ numpy.log()  ~ numpy.log1p()
~ numpy.exp()  ~ numpy.abs()
```

The Mathematical Formula That Acts as A Foundation for Nonlinear dynamic equilibrium On All Of those Modules Were:
~ Anisotropy
~ Entropy
~ Kullback-Leibler (KL) Divergence
~ Curvature Geometry
~ Nested Exponentials
~ Efficient KL comparatives
~ sigmoid 


.- Explanation About Why I Used That Mathematical Principles is How They Can Calculate Logits Or Probabilities Sensitivity, Meta Simulations Or Planner
, and compare them directly with each Divergence formulas from the Meta Simulations and the Raw Logits.
- Below is A Compact Explanation:

●. Anisotropy:
anisotropy means a phenomenon or data property changes with direction, in this case, The Property of A Matrix that changes with direction. Anisotropy Here is used to calculate the smoothness of Complexity, on how The environment the AI Faced has Changed.

~ Equations in code form:
```math
gradient = np.gradient(array)
calibration = [np.linalg.norm(v) for v in gradient]
anisotropy = np.std(calibration) / np.mean(calibration)
```

●. KL Divergence:
KL Divergence is Used To Calculate sensitivity on How much the Logit is shifting from uniform logit.
- Code Formula:
```math
Kl_divergence = np.sum(logit * np.log(logit)) - np.log(uniform_logit))
```
From the code, This Version Of KL divergence was much more sensitive to How much it has diverged, using ``` log(uniform)``` tells the model how much it has diverged, This Formula was Proved to be more Numerically stable and Efficient At Calculating Divergence from uniformity.

●. Logistic Growth equation:
A. CellularAutomataNet Used A Logistic equation to maximize Logistic Growth and Potentially Maximize Growth efficiency from Superlinear Manifold with Dynamic Constraint From Logistic equation:

``` Logistic satisfiability = 1.0 + Logistic_stability / sigmoid```

Where sigmoid creates a bounded Superlinear growth and stability ensures Logistic constraint.

B. 3 fundamental superlinear with bounded logistic growth constraint equation inspired from riemannian geometry	with differential calculus that forms a logarithmic coupling.
This equation created a stable superlinear with logistic cnstraint that was ensured from the 1.0 as statistical coupling to ensure finite distances given any logits, thus creating a differential calculus inspired from riemannian optimization and logarithmic coupling where each equations complete each others weaknessess.  

```
fundamental_logistic_geodesic= optimum / 1.0 + logistic_stability
    fundamental_logistic_stability = logistic_equilibrium / 1.0 + optimum
    fundamental_logistic_sequencing = optimum / 1.0 + sigmoid
```


C. 3 fundamental logistic equation derived to acquire the thorough geodesic info per using calculus variations that used to acquire dimensionless number of probabilities to acquire a stable modelling and a high efficiency of geodesic information in any dimensionless geometric space of moduli space properties.		   
1. (1/2) was used to calculate the moduli space of the phase projection of geometric properties of dimensionless matrix that will thoroughly acquire a stable geodesic efficiency of an information transport or data where trA2 > 0 given positive logits  to ensure geodesic stability of each logistic growth.
2. (1/6) was used to calculate the theoretical geodesic space of information efficiency through euclidean range in moduli space in respect to geometric properties where trA3 > 0 to ensure efficient search through superlinear growth with logistic constraint modelling of phase projection of any given valid value.
			    
3. simplified moduli space equations combined with geodesic mapping efficiency to ensure both logistic and superlinear growth to maximize information gather efficiency and stability ensuring both appear in geometric efficiency through moduli space search with trA3 > 0 and range 0 -> finite numbers with any given positive logits, this equation will provide implicit eigenvalues to the model (as shown in the geodesic_optimum) that can map any geodesic topological space where it will ensure the models stability and convergence.

		    
```
trA1 = projection / (1.0 - slope)
   trA2 = (1/2) + stability_modelling / 1.0 + trA1**2
   trA3 = (1/6) + logistic1 / (trA2**2) - 1.0
geodesic_optimum = np.dot(x, trA3)
```
			    


●. Curvature:
Curvature Is Used to calculate the geometry curve of the logits and the curvature of each Nested logit or Probabilities Simulation.

- Code Formula:
```math 
curvature = np.mean(np.abs(np.diff(np.diff(logit))))
```

• From The Code formula, ```
numpy.mean()``` is used to calculate mean on logits itself directly. While For ```numpy.abs()``` itself is to turn each scalar inside the list to be an absolute value, so double differential scaling will be much easier and precisely accurate, the double ```numpy.diff()``` is used to calculate the differential value of each scalar inside that logit, double usage here is used to achieve precision higher.

●. Nested Meta Simulations:
Nested meta Simulations here Are A bunch of ```numpy.exp()``` scaling that exponentiate logits to see how the model can simulate logit by scaling it to predict how future logits looked like.

●. Hitchins Moduli Space:
Hitchins moduli space is a mathematical equation to calculate the theoretical geometry of each matrix in this case, the equation codes:

```
(example from CellularImbrium class)
				alpha1 = np.dot(x, multipliers)
				alpha2 = np.dot(x, mutation)
				alpha3 = np.dot(x, equilibrium)	
							
				trA1 = np.linalg.norm(alpha1)
				trA2 = np.linalg.norm(alpha2)
				trA3 = np.linalg.norm(alpha3)	
				s1 = trA1**2 - trA3 / equilibrium
				s2 = (1/2) * (trA2**3 + trA3** 2  / equilibrium)
				s3 = (1/6) * (trA3**3 - (3 * trA1 * trA2**2) + (2 * trA3**3) / (3 * trA3**3) / equilibrium)
				all_sample_scores = (1.0 + s1 + s3 / s3 + s2 - s1)
```

where trAa1 used to calculate th the magnitude of a vector or matrix, and seasons (the s1, s2, s3 part) will calculate the theoretical geometrical position based on the magnitude of that matrix or a vector by dividing it with equilibrium, it ensures each seasons can theoretically return a stable equilibrium scalar.



