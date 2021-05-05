# this mod is all the data_processed function included



#this funciton could suffix all the file needed
def suffix(file, *suffixName):
    array = map(file.endswith, suffixName)
    if True in array:
        return True
    else:
        return False

#source ,target is the file_fold
#in this data cleansing function we mainly extrcat all the file
#which is endwith .outmol
def extractFile(source:string,target:string,endwith='*.outmol'):
#sourceDIr = eachFIle(filePath)
    sourceDir = source
    targetDir = target
    for root, dirs, files in os.walk(source):
        for name in files:
            if fnmatch.fnmatch(name,endwith):
                sourceFile = os.path.join(root,name)
                shutil.copy(sourceFile,targetDir)


#this three funciton is only design for our raw_data
#for we have explore these data feature and have applied data cleansing
def extractCoordinates(file:file_obj):
    _file = file.read()
    start_1 = file_obj.find('$coordinates')
    end_1 = file_obj.find('$end')
    return _file[start_1+len('$coordinates'):end_1]


def extractEnergy(file:file_obj):
    _file = file.read()
    start_2 = file_obj.find('Total Energy')
    end_2 = file_obj.find('Message: SCF converged')
    str_energy = _file[start_2:end_2]
    return str_energy


def extractData(file_dir:string,target_dir:string):
    for file_name in os.listdir(file_dir):
        file = file_dir + file_name
        f = open(file,encoding='gb18030',errors='ignore')
        file_obj = f.read()
        print(file_name)

        start_1 = file_obj.find('$coordinates')
        end_1 = file_obj.find('$end')
        start_2 = file_obj.find('Total Energy')
        end_2 = file_obj.find('Message: SCF converged')
    # redd the location of the coordinate and converged energy

    if start_2 != -1 and end_2 != -1 and end_2-start_2 >=5:
        str_coordinates = file_obj[start_1+len('$coordinates'):end_1]
        str_energy = file_obj[start_2:end_2]
        #read data

        energy = str_energy.split()[-5]
        result = str_coordinates.split()

        result.append(float(re.findall(r'\d*\.\d*',energy)[0]))
        np.savetxt(target_dir + file_name+'.csv',result,fmt='%s')
