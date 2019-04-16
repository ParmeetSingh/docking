import socket
import time

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
#server = socket.socket(socket.AF_INET, # Internet
#                        socket.SOCK_DGRAM) # UDP
# Set a timeout so the socket does not block
# indefinitely when trying to receive data.
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(0.2)
message = b"your very important message"
while True:
    server.sendto(message, ('<broadcast>', 51110))
    print("message sent!")
    time.sleep(1)