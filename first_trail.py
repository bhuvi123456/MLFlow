import mlflow
def calculate_sum(x,y):
    return x+y

if __name__ == '__main__':
    with mlflow.start_run():
        x,y = 20,30
        z = calculate_sum(x,y)
        # To track the data
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)