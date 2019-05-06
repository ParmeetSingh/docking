echo nvidia | sudo -S ifconfig eth0 192.168.214.41 up
cd ~/docking
python3 docking-identification_yolo_weights_inputv2.py -w weights/yolo_mobilenet_weights_v11_v2.hd5 -cfg cfg/yolo_mobilenet_weights_v11_v2.cfg -slink http://192.168.2.54:8080/video -s big -l True -conn_test True -m air -skip 10 -p1 51110 -p2 51120