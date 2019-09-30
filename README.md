# CapstoneDesign

Seoultech 2019 Electronic and IT Media Engineering Capstome Design

Attendance check server & client

* VER. Release.
* Overall Framework : API Server (Flask) + MTCNN + ArcFace (Pytorch 0.4.1)
* Face recognition using API Server.


## Turn on the server (using T-mux)

1. SSH 를 통한 서버 접속.  

2. 터미널 창에 `tmux attach -t server`  를 입력해서, server 세션에 접속합니다.  
  
3. 패키지가 설치된 아나콘다 가상환경을 실행시켜줍니다. 

   ```
   source activate face-recog
   ```

4. ~/face-server/dl 디렉토리로 이동하여 run_server.py 를 실행시켜주시면 됩니다.  

   ```
   cd ~/face-server/dl
   python run_server.py
   ```

## Environment Setting

**Virtual environment (using Anaconda)**

1. 아나콘다 설치 이후, `conda create -n [가상환경이름] python=3.6` 을 사용하여 가상환경을 생성해줍니다.  
  
2.  가상환경을 실행시켜 줍니다.  
   `source activate [가상환경이름]`   

3. Pytorch를 설치 해줍니다.  
   `conda install pytorch=0.4.1 torchvision -c pytorch `   

4. Python 패키지들을 설치 해줍니다.  
   `pip install -r requirements.txt`  

5. 사용이 종료되면 가상환경을 종료 해줍니다.  
   `source deactivate` or `deactivate`



**T-mux**

1. T-mux 세션 생성 (세션 외부에서)  
   `tmux new -s [세션이름]`  

2. T-mux 세션 종료 (세션 외부에서)  
   `(tmux 세션 상에서) exit `  

3. 세션 접속하기 (세션 외부에서)  
   `tmux attach -t [세션이름]`  

4. 세션에서 나오기 (세션 내부에서)  
   `ctrl + b, d`  

5. T-mux 세션 리스트 확인하기 (세션 외부에서)  
   `tmux ls`



