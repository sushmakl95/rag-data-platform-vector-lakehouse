pipeline {
    agent { docker { image 'python:3.11-slim' } }
    options {
        timeout(time: 15, unit: 'MINUTES')
        timestamps()
    }
    stages {
        stage('Setup') {
            steps {
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements-dev.txt'
            }
        }
        stage('Lint') {
            steps {
                sh 'ruff check .'
                sh 'black --check .'
                sh 'yamllint .'
            }
        }
        stage('Tests') {
            steps {
                sh 'pytest tests/unit -v --junitxml=reports/junit.xml'
            }
            post { always { junit 'reports/junit.xml' } }
        }
    }
}
