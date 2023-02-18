url="https://drive.google.com/file/d/1LdjOv8XIlu8-GTL_n-yySF3B-rPNEBhH/view?usp=share_link"
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
size="width=\"\" "+"height =\"\" "
tag="<img src='"+path+"' "+size+"/><br>"

url.split('/')
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
print(" ▶ Path : ", path)