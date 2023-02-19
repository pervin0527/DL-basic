def convert_url(url):    
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    size="width=\"\" "+"height =\"\" "
    tag="<img src='"+path+"' "+size+"/><br>"

    url.split('/')
    path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    print(" ▶ Path : ", path)

input_url = "https://drive.google.com/file/d/1OWpDF4PlSOnyGfLBQYI3V8RB8FW9okp2/view?usp=share_link"
convert_url(input_url)