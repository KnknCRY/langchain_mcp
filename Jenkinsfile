pipeline {
  agent any

  stages {
    stage("build image") {
      steps {
        script {
            withCredentials(
                [usernamePassword(credentialsId: 'docker-hub',
                usernameVariable: 'DOCKER_HUB_USERNAME', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                sh "docker build -t s35016080/python-mcp:3.0 ."
                sh "echo $DOCKER_HUB_PASSWORD | docker login -u $DOCKER_HUB_USERNAME --password-stdin "
                sh "docker push s35016080/python-mcp:3.0"
            }
        }
      }
    }

    stage("deploy") {
      steps {
        script {
            echo "deploy"
        }
      }
    }
  }
 
}