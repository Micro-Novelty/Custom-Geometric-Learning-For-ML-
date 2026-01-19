
# ●. Beginner Guides and Introduction:
A. Neuro-Symbolic AI (NSAI) is a hybrid AI approach that merges the pattern recognition strengths of neural networks (deep learning) with the logical reasoning of symbolic AI (rules, knowledge graphs) to create more robust, explainable, and human-like AI systems, tackling limitations like AI hallucinations by grounding data-driven learning in explicit knowledge. 

●. CronicalSpark Represents an architectural shift in how Neuro Symbolic AI Used Geometric Learning to reasons about causality and context correlation, This kind of AI usually rarely converged for longer reasoning Epochs, because it differs by the noise and divergence of continous complexity in an environment, thus making it reliable for Real world Complex environment. The Reason why this Non-convergence behavior is valuable is because Real World environments has Complex noise that consist of micro to macro change in each pattern, thus Longer convergence can't emerge randomly for Neuro Symbolic AI because the output was determined from the consistent internal coherences of its pattern recognition match and reasoning, thus a model must adapts and continously learn each patterns and correlations based on the complexity of situation, so an AI that Doesnt Converge And Overfits is a practical Minimum for real world deployment in which current Deep learning Lacks.

# 🌐 Key Features You can try!:
CronicalSpark is not regular Deep learning, so to visualize its Internal Reasoning, we need to use:
1. Kurtosis and Skew histogram
2. Power Law Tail
3. Autocorrelation Function
4. Return Map and Q-Q Plot 

●. This Functions will enable us to see the Internal dynamic of How CronicalSpark can reasons when it was given a random input, and forms a stable regime of reasoning, you can Test this yourself, i provided the codes for This kind of testing so you can See and expand the test!

# 📃 Small Note:
This Branch Contained Explanation regarding CronicalSpark AI, Also I Have Provided A TestBed for Cronical Spark itself that anyone can use and see the Performance of CronicalSpark 
Note = You can immediately execute the  "domain test" code of CronicalSpark without any modification, the results will show a matplot of graphics that correlates to CronicalSpark abilities.
*. License = MIT

# 📑 Libraries in Python Requirements:
1. Numpy for processing
2. Matplotlib for visualization
• Supports Python 3.8+


~ Hope you understand the math and its Functions. Have Fun Checking and testing, and Feel Free to suggest improvements and Forks! If you want further Information regarding an Empirical results, i've already provided some explanation and results plots that you can immediately see and conclude in discussions. :)

# ●. Technical Explanation Of CronicalSpark:
●. CronicalSpark is a Custom Complex Reasoning AI That Was Using Certainty as a Measurement of its Performance on task, This AI Performs A Good Certainty (without being overconfident swinging up down like Regular AI on little trainings)  In Linear And Sine Pattern Recognition, While Also Being Strategically Cautious and Opportunist At Random and Step Pattern. 

●. This AI can also Have Dual Mode, The Human in the Loop Intervention And Autonomous Mode, There is a dedicated Function for CronicalSpark Specifically For It if CronicalSpark needed Human judgement for feedback because it faced a novel uncertainty of conditions. 

●. IMPORTANT NOTES:
~ This AI doesnt have explicit training functions, it can train itself based on the reasoning it has done before and modify its memory values, including weight, bias and reasoning batch. The quality of training depends on the coherence belief of its previous stage and Current stage of reasoning. This allows it to prevent overfitting, because almost all its weight, bias and reasoning batch has its own value of complexity from the sensitive_sigmoid and anisotropy (later explained in the mathematical explanation below).

It only has a small training module in its sub_agent function, dedicated to make the sub agent to cooperate on consensus. But it doesnt influenced the entire reasoning architecture, only the local sub_agent.


●. Core capabilities:
1. One of the key capabilities of CronicalSpark is that it can reason epistemically:

The small_predictive_embedding_module is used for reasoning that contains many functions, Specifically for simulative scenario like counterfactual probe, and factual concrete reasoning to form reasoning behavior and later used to correlates its quality of Reasoning.

2. It has built in memory storage and can do memory overriding if needed.

Once it receives its first input, the AI will learn to used and manipulate it in order for reasoning for future possibilities. the memory can store up to hundreds of thousands (you can increase it by overwriting the computational_limit = 100.000) so it can store a lot of memory, if youre wondering if this AI can judge all that memory without it get wrong, is that each weight and bias were calculated using sensitive_sigmoid so that each pair of Array of weights and bias is impossible to have the same value of sensitive_sigmoid. While The raw reasoning batch (specifically after the model has stored its weights and bias) is calculated and stored with the anisotropy value of each matrix that correlates with the environment.

From this capabilities, the AI can learn to specialize randomly (without hardcoding) based on its first input it has faced, in human terms this is like internal "belief", or intuition.

3. The model will learn to correlates conditions and form causal modelling of its internal state based on the external feedback (input or human judgement):

This capabilities can be achieved if the AI has learned or trained itself quite a while,the internal_causal_modelling function will allow it to correlate its reasoning batch with weights batch to form causality correlation, and after that, it will correlate it with reasoning module of simulative_search in predictive _embedding_module to further push its reasoning judgement and correlation or causality quality.

3. Low Computing time:

Computing time for this AI usually sits around 50 ms and peak around 100 - 200 ms for deeper reasoning, can be made efficient, but for now, all is numpy because its readable and provides easier modification in the future.

# 📌 Quick start to use CronicalSpark:

1. First Initialize CronicalSpark class, 
Assign the computational limit (this mean s the memory limit of CronicalSpark, ex =1000 memory limit size) and your desired output size (ex. = 10 or 20 values in 1D matrix, depends on your output size you want), and the last is processing_type, wether you want the output equal or extended or less. By assigning "equal_input_output" in processing_type, CronicalSpark will prioritize the output to be the same shape and size as the input, regardless of the desired_output you assigned.
Else other than "equal_input_output", will make the output size different than the input size, matching your assigned  desired_output size.

(size here means each value inside a matrix if a matrix consist of 3 values :
Ex = [1, 2, 3], its size will be 3]

2. Next, Find meta_definitor(x) function, where x is a 1D or 2D Matrix as the first input for CronicalSpark to process that consist of multiple scalars inside the matrix.

●. IMPORTANT CLARITY=

~ If "equal_input_output" means that CronicalSpark Input and output is equal, meaning that the number of Input will be the same for output, meaning if you want to extract the max value inside a matrix output, you need to use numpy.argmax(), so actions or tasks must be encoded first in a Python list. The number of total scalars inside a matrix depends on your needs and CronicalSpark doesn't have any restrictions in total scalars of an input. 

●. Example Usage:
```math
Example_list = 
["apple", "orange", ....]
input = [0.05, 0.002, ...]
CronicalSpark_output = [0.2, 0.7, ...] or meta_definitor(input)
max = np.argmax(CronicalSpark_output)
Get_Max_value = Example_list[max]
```


# ⚡ Mathematical Foundation And Expression

The Mathematical Principle used to built CronicalSpark used Numpy syntaxes Such as:

```math
~ numpy.log()  ~ numpy.linalg.norm()
~ numpy.exp()  ~ numpy.abs()
~ numpy.mean() ~ numpy.std()
```

The Mathematical Formula That Acts as A Foundation for Nonlinear and linear dynamic equations (derived from first principles) On All Of those Modules in CronicalSpark Were:
~ Anisotropy
~ Entropy
~ Kullback-Leibler (KL) Divergence
~ Curvature Geometry
~ sigmoid and sensitive_sigmoid 
~ 3 logistic equations (derived from Riemannian geometric equations)


●- Explanation About Why I Used That Mathematical Principles is How They Can Calculate Logits Or Probabilities Sensitivity, Meta Simulations Or Planner
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

Where np.linalg.norm is used to calculate the magnitude of each value inside the gradient to be used for calculating the complexity of the array it face.

●. sigmoid or sensitive_sigmoid:
~ sigmoid is used to calculate the non linear dynamic of which a matrix that has an anisotropy above > 0.5 and a inconsistent divergence in how each values differs in gradient form. 

● equations =
```math
sigmoid = 1.0 / (1.0 + curvature)
```

~ sensitive_sigmoid is used to calculate the minimum sensitivity of how a matrix (x) can reach from f(x) = x < 0 and x > 0, where its a useful function to calculate the sensitivity of each non linear pattern by acquiring the minimum value of the sensitivity of non linearities from the matrix value.

● equations =
```math
sensitive_ sigmoid = 1.0 / (1.0 - curvature)
```

●. Entropy:
Entropy is used to calculate the initial loss of value a matrix has. The use of Entropy here is to minimize a decent loss of information preservation when calculating the Geometric value of a matrix.

~ Equations in code form:
```math
Initial_value = x1 / np.sum(x1)
distance = Initial_value[Initial_value > 0]	
entropy = -np.sum(distance * np.log2(distance))
```

Here distance is used to prevent negative value in order to prevent underflow.



C. 3 fundamental logistic equation derived to acquire the thorough geodesic info per using calculus variations that used to acquire a dimensionless number of probabilities to acquire a stable modelling and a high efficiency of geodesic information in any dimensionless geometric space of properties in moduli space. 

●. Explanation of components:	
   
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
numpy.mean()``` is used to calculate mean on logits inside a matrix itself directly. While For ```numpy.abs()``` itself is to turn each scalar inside the list to be an absolute value, so double differential scaling will be much easier and precisely accurate after numpy.abs(). the double ```numpy.diff()``` is used to calculate the differential value of each scalar inside that matrix, double usage here is used to acquire a second order curvature value for an accurate Probability of a curvature a matrix has.


# 🌐 Specific Use Case:

1. Game Agent:
This AI can be plugged easily to any Game that requires An Agent to reason in a complex environment without it being overfitting, it only needs one external functions that the AI wants, an external functions that provides a human judgement, Specifically for context needs if the AI is very uncertain, but can made Autonomous too if you don't want any human intervention by only providing the correct answer to a logits form automatically, or, also a shared value of other CronicalSpark instance for multi agent Cooperation, in the external function. All depends on your needs.

2. General Domain (hypothesized Only, but Empirical results are promising):
Specifically for Complex Dynamic of finance and markets, that can detect a market change, and calculates the future consequences based on the market and change of needs complexity.
CronicalSpark pattern recognition certainty on linear and non linear dynamic is also viable for Medical diagnosis if viable.
●~ but General Domain Usage will require an extensive Test of real deployment and red teaming, so this hypothesis can be cut off if unproven, but the previous test is already promising because it can perform well in certainty linear, and sine pattern, while also being strategically cautious in random and sine pattern. can also be extended to task-reliability domain. 
●. This Was only a hypothesized capabilities, but Empirical results Showed CronicalSpark can reasons for long term conditions, and evaluate it based on certainty metrics and groundedness of alignment with internal coherences and external stimuli for anisotropy based on the histogram of plot results in discussion, but further tests are required.

# Development Credit to Author:

●. Author = Anonimity (X.11) / Indonesia As part of a standalone research Project for Studying How Neuro Symbolic AI could Be Programmed using equations derived from Geometric Riemannian equations and Advanced Programming logic for efficient Execution for reasoning capabilities.


