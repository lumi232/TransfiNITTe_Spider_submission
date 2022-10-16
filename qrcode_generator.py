import qrcode 
img = qrcode.make("http://192.168.4.1/L")
img.save("wm2.jpg")

