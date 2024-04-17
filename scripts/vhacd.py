
from os import listdir,mkdir
from os.path import isfile, join
import subprocess

mypath = "object_models/bags"
onlyfiles = [f for f in listdir(mypath)]

# Update "/path/to/bullet3/" part of the subprocess
for x in onlyfiles:
    obj_file = join(x, "model.obj")
    new_obj_file = join(x, "model_vhacd.obj")  
    subprocess.call(["/path/to/bullet3/bin/test_vhacd_gmake_x64_release",'--input', join(mypath,obj_file),'--output', join(mypath,new_obj_file)])
    
