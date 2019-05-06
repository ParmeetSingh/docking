import tensorflow as tf
from utils import Message,Message2
import socket,sys


try:
	avail = tf.test.is_gpu_available(cuda_only=True)
except:
	print("GPU Not available")
	avail = False		
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(0.2)

if avail==True:
	print("GPU available")
	sys.exit()
	
while avail==False:
	m1 = Message()
	m2 = Message()

	m1.set_date_time()
	m1.set_camera_status(-1)
	m1.set_checksum()

	m2.set_date_time()
	m2.set_camera_status(-1)
	m2.set_checksum()
	logger.info("Message 1 is %s",m1.convert_to_string())
	logger.info("Message 2 is %s",m2.convert_to_string())

	print("Sending m1 to port:"+port1)
	server.sendto(m1.convert_to_string().encode(), ('<broadcast>', int(port1)))
	print("Sending m2 to port:"+port2)
	server.sendto(m2.convert_to_string().encode(), ('<broadcast>', int(port2)))
	try:
		avail = tf.test.is_gpu_available(cuda_only=True)
	except:
		print("GPU Not available")
		avail = False		

sys.exit()		
