cd posture-monitor-main
pip install -r requirements.txt
python3 -m pip install --upgrade pip
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
pip uninstall opencv-python -y
pip install opencv-python-headless
sudo apt-get update && sudo apt-get install -y libasound2-dev
pip install simpleaudio
sudo apt update && sudo apt install ffmpeg -y