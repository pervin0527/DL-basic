url="https://drive.google.com/file/d/141jLWopFztKAOzOEiKwDYBlFwFig4bI9/view?usp=share_link"
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
size="width=\"\" "+"height =\"\" "
tag="<img src='"+path+"' "+size+"/><br>"
# print(" ▶ Path : ", path)
# print('\n',"▶ Tag : ", tag)

url.split('/')
path='https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
print(" ▶ Path : ", path)