def convert_url(url):    
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    size="width=\"\" "+"height =\"\" "
    tag="<img src='"+path+"' "+size+"/><br>"

    url.split('/')
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    print(" ▶ Path : ", path)

input_url = "https://drive.google.com/file/d/1-WsqF4J4CIDSgaZooBjnx-s3hP3Yu8Db/view?usp=share_link"
convert_url(input_url)