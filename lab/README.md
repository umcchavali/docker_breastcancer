##  🏥 Breast Cancer Dataset: A Simple Introduction 

https://archive.ics.uci.edu/dataset/14/breast+cancer
👋 This dataset is a hypothetical representation of 200 breast cancer patients, designed for learning purposes. 💡 It includes relevant information like age, tumor size, lymph node involvement, and treatment, along with the diagnosis of breast cancer (yes or no). 

**📊 What's in the Data?**

* 👴 age: Age of the patient at diagnosis (in years) 
* 👵 menopause: Menopausal status (premeno, lt40, ge40) 
* 📏 tumor_size: Size of the tumor (in mm) 
* 📍 inv_nodes: Number of involved lymph nodes 
* 🛡️ node_caps: Presence of node capsular invasion (yes/no) 
* 🧬 deg_malig: Degree of malignancy (1-3) 
* 左右 breast: Breast affected (left/right) 
* 🧭 breast_quad: Quadrant of the breast where the tumor is located (left_up, left_low, right_up, right_low) 
* ☢️ irradiat: Whether the patient received radiation therapy (yes/no) 
* 🎗️ target: The outcome variable - whether the patient has breast cancer (1) or not (0) 

**⚠️ Important Notes:**

* 🤔 This is not real patient data and should not be used for medical decision-making. It's a simplified representation for learning purposes. 
* 🤖 Categorical features are encoded using one-hot encoding, which means each unique category value gets its own binary column. 
* 🧮 This dataset is set up for running a linear regression model, but remember that breast cancer diagnosis is a classification problem.  
* This dataset is an inspiration from cdc and kaggle. I then used LLM's to make a clean dataset for project purposes.
* Example: https://github.com/NanditaA/ML-Supervised-Classification/blob/master/README.md 

**🚀 How to Use This Dataset:** 

* 💻 Practice Linear Regression: Build and evaluate a linear regression model to predict the probability of breast cancer diagnosis based on the given features. 
* 🛠️ Data Preprocessing:  Practice one-hot encoding and explore other data preprocessing techniques.
* 📈 Model Evaluation:  Learn how to evaluate model performance using metrics like MSE. 

🎉 Have Fun Learning!

🤝 Remember to consult real breast cancer datasets and research for more accurate and complex analysis. Be mindful of ethical and privacy considerations when working with sensitive medical data.

# Run the Code:

"""
```
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Create a SparkSession
spark = SparkSession.builder.appName("BreastCancerTransformation").getOrCreate()

# Load Data and Create DataFrame
dataset = spark.read.csv(
    "/mnt/stodsba6190delta/breastcancer.csv",
    header=True, 
    inferSchema=True  # This will automatically infer the data types
) 

label = "target"
categoricalColumns = ["menopause", "node_caps", "breast", "breast_quad", "irradiat"] 
numericalColumns = ["age", "tumor_size", "inv_nodes", "deg_malig"] 

categoricalColumnsclassVec = [c + "classVec" for c in categoricalColumns]

# Feature transformation steps
stages = []

for categoricalColumn in categoricalColumns:
  print(categoricalColumn)

  stringIndexer = StringIndexer(inputCol=categoricalColumn, outputCol = categoricalColumn+"Index").setHandleInvalid("skip")

  encoder = OneHotEncoder(inputCol=categoricalColumn+"Index", outputCol=categoricalColumn+"classVec")

  stages += [stringIndexer, encoder]

# Create the VectorAssembler
assembler = VectorAssembler(
    inputCols=[
        "age",
        "tumor_size",
        "inv_nodes",
        "deg_malig",
        "menopauseclassVec",
        "node_capsclassVec",
        "breastclassVec",
        "breast_quadclassVec",
        "irradiatclassVec"
    ],
    outputCol="features"
)

# Add the assembler to the stages
stages += [assembler]

# Feature Scaling with StandardScaler
scaler = StandardScaler(inputCol = "features",
                        outputCol = "scaledFeatures",
                        withStd = True,
                        withMean = True)
stages += [scaler]
"""

"""
# Create the Pipeline
pipeline_uma = Pipeline(stages=stages)

# Fit the pipeline to your data
pipelineModel = pipeline_uma.fit(dataset)

# Delete the existing directory if it exists
#dbutils.fs.rm("/mnt/stodsba6190delta/pipeline_uma", True)

# Save the pipeline model (overwrite if it exists)
#pipelineModel.write().overwrite().save("/mnt/stodsba6190delta/pipeline_uma") 

# Display the saved pipeline
display(dbutils.fs.ls("/mnt/stodsba6190delta/pipeline_uma"))

# Load the saved pipeline model
pipelineModel = PipelineModel.load("/mnt/stodsba6190delta/pipeline_uma") 

# Apply transformations to a new dataset 
dataset = spark.read.csv(
    "/mnt/stodsba6190delta/breastcancer.csv",
    header=True,
    inferSchema=True  # Important!
) 
transformed_dataset = pipelineModel.transform(dataset)

# Show transformed data
display(transformed_dataset)

"""
# Log Reg
"""
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Split the transformed data into training and testing sets
(trainingData, testData) = transformed_dataset.randomSplit([0.7, 0.3], seed=12345)

# Create the logistic regression model
#lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=label)

lr = LogisticRegression(featuresCol="scaledFeatures", labelCol=label, 
                        regParam=0.1,  # Adjust this value
                        elasticNetParam=0.5) # Adjust this value

# Define the parameter grid for hyperparameter tuning
paramGrid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.1, 0.01])
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
    .build())

# Define the evaluators for model evaluation
binaryEvaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
multiclassEvaluator = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")

# Create the cross-validator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=binaryEvaluator, numFolds=5)

# Fit the cross-validator to the training data
cvModel_lr = cv.fit(trainingData)

# Evaluate the best model on the test data
predictions = cvModel_lr.transform(testData)
auc_lr = binaryEvaluator.evaluate(predictions)
accuracy_lr = multiclassEvaluator.evaluate(predictions)

# Print the AUC, accuracy, and other metrics
print("Test AUC:", auc_lr)
print("Test Accuracy:", accuracy_lr)
"""

"""
# Decision Tree

from pyspark.ml.classification import DecisionTreeClassifier
# Split the transformed data into training and testing sets
(trainingData, testData) = transformed_dataset.randomSplit([0.7, 0.3], seed=12345)

# Create the Decision Tree model
dt = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol=label)

# Define the parameter grid for hyperparameter tuning
paramGrid = (ParamGridBuilder()
    .addGrid(dt.maxDepth, [5, 10, 15])  # Experiment with different depths
    .addGrid(dt.minInstancesPerNode, [1, 5, 10])  # Minimum samples at each node
    .build())

# Define the evaluators for model evaluation
binaryEvaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
multiclassEvaluator = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")

# Create the cross-validator
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=binaryEvaluator, numFolds=5)

# Fit the cross-validator to the training data
cvModel_dt = cv.fit(trainingData)

# Evaluate the best model on the test data
predictions = cvModel_dt.transform(testData)
auc_dt = binaryEvaluator.evaluate(predictions)
accuracy_dt = multiclassEvaluator.evaluate(predictions)

# Print the AUC, accuracy, and other metrics
print("Test Tree AUC:", auc_dt)
print("Test Tree Accuracy:", accuracy_dt)

"""

from pyspark.ml.param import Param, Params

print("Logistic Regression - Test AUC:", auc_lr)
print("Logistic Regression - Test Accuracy:", accuracy_lr)
print("Decision Tree - Test AUC:", auc_dt)
print("Decision Tree - Test Accuracy:", accuracy_dt)

if auc_lr > auc_dt:
    print("Logistic Regression performed better based on AUC.")
    best_model = cvModel_lr 
    best_params = best_model.bestModel.extractParamMap()
elif auc_dt > auc_lr:
    print("Decision Tree performed better based on AUC.")
    best_model = cvModel_dt 
    best_params = best_model.bestModel.extractParamMap()
else:
    print("Models performed similarly based on AUC.")
    # Choose one model as the best model, e.g., Logistic Regression
    best_model = cvModel_lr 
    best_params = best_model.bestModel.extractParamMap()

#print("\nBest Model Parameters:")
#for param in best_model.bestModel.params:
 #   print(f"{param}: {best_model.bestModel.getOrDefault(param)}")

"""

"""
# Print best model parameters
print("\nBest Model Parameters:")
# Ensure 'lowerBoundsOnCoefficients' is in the model's parameters
if 'lowerBoundsOnCoefficients' in best_model.bestModel.params:
    for param in best_model.bestModel.params:
        print(f"{param}: {best_model.bestModel.getOrDefault(param)}")
else:
    print("'lowerBoundsOnCoefficients' parameter not found in the model.")

"""
"""
print(
    "\nBest Model Parameters:",
    *[
        f"{param.name}: {best_model.bestModel.getOrDefault(param)}"
        for param in best_model.bestModel.params
        if best_model.bestModel.hasDefault(param)
    ]
)
"""

# Delete existing directories
#dbutils.fs.rm("/mnt/stodsba6190delta/pipeline_uma/", True) 
#dbutils.fs.rm("/mnt/stodsba6190delta/models/", True)

# Save pipeline and model
#pipelineModel.write().overwrite().save("/mnt/stodsba6190delta/pipeline_uma/")
best_model.write().overwrite().save("/mnt/stodsba6190delta/models/")

#only run the commands once or else slight errors due to code repetition (multiple saves)

# End of Assignment - Uma Chavali
"""
