import mlflow

# WHERE MLflow saves everything: a local database file in your project folder 
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# WHICH experiment (folder). Created automatically if it doesnt exist
mlflow.set_experiment("hello-mlflow")

#  one "run" = everything inside this with-block
# with mlflow.start_run(run_name="first_run"):
with mlflow.start_run(run_name="second_run"):
    # Parameters: settings, logged once 
    # mlflow.log_param("learning_rate", 0.001)
    # mlflow.log_param("epochs", 5)
    # i change learning_rate to 0.01 and run_name to "second-run" so can see what is the different
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
     
    # Metrics: logged repeteadly, so MLflow can draw a curve 
    for epoch in range (1,6):
        fake_loss = 1.0/epoch #pretend the loss improves each epoch 
        mlflow.log_metric("fake_loss", fake_loss, step=epoch)

    # Artifcat: a file attacged to the run 
    mlflow.log_text("This is my first MLflow run!", "note.txt")

print("Done. Run recorderd.")