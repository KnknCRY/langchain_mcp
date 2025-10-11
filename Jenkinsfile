def gv

pipeline {
  agent any

  stages {
    stage("init") {
        steps {
            script {
                gv = load 'script.groovy'
            }
        }
    }
    
    stage("build image") {
      steps {
        script {
            gv.buildImage()
            // withCredentials(
            //     [usernamePassword(credentialsId: 'docker-hub',
            //     usernameVariable: 'USER', passwordVariable: 'PASS')]) {
            //     sh "docker build -t s35016080/python-mcp:3.0 ."
            //     sh "echo $PASS | docker login -u $USER --password-stdin "
            //     sh "docker push s35016080/python-mcp:3.0"
            // }
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