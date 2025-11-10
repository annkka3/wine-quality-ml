from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="wine_quality_train",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 3 * * *",  # ежедневно в 03:00
    catchup=False,
) as dag:

    dvc_pull_data = BashOperator(
        task_id="dvc_pull_data",
        bash_command="cd /opt/airflow/dags/repo && dvc pull data/winequality-red.csv",
    )

    train_rf = BashOperator(
        task_id="train_rf",
        bash_command="cd /opt/airflow/dags/repo && python -m src.models.train --model random_forest",
    )

    train_gbrt = BashOperator(
        task_id="train_gbrt",
        bash_command="cd /opt/airflow/dags/repo && python -m src.models.train --model gbrt",
    )

    select_best = BashOperator(
        task_id="select_best",
        bash_command="cd /opt/airflow/dags/repo && python -m src.models.select_best",
    )

    dvc_push_model = BashOperator(
        task_id="dvc_push_model",
        bash_command="cd /opt/airflow/dags/repo && "
                     "dvc add models/model_best.pkl models/model_metadata.json && "
                     "git add models/model_best.pkl.dvc models/model_metadata.json.dvc && "
                     "git commit -m 'Update best model and metadata' || echo 'No changes' && "
                     "dvc push",
    )

    dvc_pull_data >> [train_rf, train_gbrt] >> select_best >> dvc_push_model
