import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Print versions
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# Load the training dataset
train_file_path = "/workspace/Housing-Sales/train.csv"
dataset_df = pd.read_csv(train_file_path)
print(f"Full train dataset shape is {dataset_df.shape}")

# Display basic info and stats about the dataset
dataset_df.head(3)
dataset_df.info()
print(dataset_df['SalePrice'].describe())

# Visualize the distribution of SalePrice
plt.figure(figsize=(9, 8))
sns.histplot(dataset_df['SalePrice'], kde=True, color='g', bins=100, alpha=0.4)
plt.show()

# Filter only numerical columns
df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
df_num.head()

# Plot histograms for numerical features
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.show()

# Split the dataset into training and validation sets
def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print(f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in testing.")

# Prepare the datasets for TensorFlow Decision Forests
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# Train a RandomForest model
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])  # Optional: Include evaluation metrics like MSE
rf.fit(x=train_ds)

# Plot a tree from the forest (this is specific to Colab)
# tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# Plot training logs (Number of trees vs RMSE)
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

# Evaluate the model on the validation set
inspector = rf.make_inspector()
evaluation = rf.evaluate(x=valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# Plot feature importances
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract feature names and importance values
feature_names = [vi[0] for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]

# Plotting the feature importances
plt.figure(figsize=(12, 8))
bar = plt.barh(range(len(feature_names)), feature_importances, color='skyblue')
plt.yticks(range(len(feature_names)), feature_names)
plt.gca().invert_yaxis()

# Annotate each bar with its importance value
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_width(), patch.get_y() + patch.get_height() / 2, f"{importance:.4f}", va="center")

plt.xlabel(variable_importance_metric)
plt.title("Feature Importances (NUM_AS_ROOT)")
plt.tight_layout()
plt.show()

# Load the test data and prepare it for prediction
test_file_path = "/workspace/Housing-Sales/test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, task=tfdf.keras.Task.REGRESSION)

# Predict on the test data
preds = rf.predict(test_ds)

# Prepare the submission file
output = pd.DataFrame({'Id': ids, 'SalePrice': preds.squeeze()})
output.to_csv('/workspace/Housing-Sales/submission.csv', index=False)

# Display the first few rows of the submission file
output.head()

# Optionally, update the sample submission
sample_submission_df = pd.read_csv('/workspace/Housing-Sales/sample_submission.csv')
sample_submission_df['SalePrice'] = preds.squeeze()
sample_submission_df.to_csv('/workspace/Housing-Sales/sample_submission.csv', index=False)
sample_submission_df.head()
