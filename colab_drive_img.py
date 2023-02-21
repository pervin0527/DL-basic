def convert_url(url):    
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    size="width=\"\" "+"height =\"\" "
    tag="<img src='"+path+"' "+size+"/><br>"

    url.split('/')
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    print(" ▶ Path : ", path)

input_url = input("type url : ")
convert_url(input_url)