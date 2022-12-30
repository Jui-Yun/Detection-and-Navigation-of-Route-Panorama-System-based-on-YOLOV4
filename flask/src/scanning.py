def Object_Detection():
    os.chdir('./yolov4_twentyclasses2/darknet')

    get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights ../mask_test_images/scan.jpg -thresh 0.3')
    get_ipython().system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights -ext_output -dont_show < /home/jcwang/YOLOV4_TRAINING/yolov4_twentyclasses2/darknet/data/scan.txt> result.txt -thresh 0.3')

    result = open('result.txt', 'r')
    index = ['administ:', 'library:', 'audio:', 'science:', 'tower:', 'exercise:', 'mingde:',
            'swim:', 'creative:',  'mainhall:', 'zhishan:', 'duxing:', 'fanglan:', 'wenhui:', 
            'studentcenter:', 'dormitory1:', 'dormitory2:', 'audioback:', 'gallery:', 
            'doorway:']
    index1 = ['administ', 'library', 'audio', 'science', 'tower', 'exercise', 'mingde',
            'swim', 'creative',  'mainhall', 'zhishan', 'duxing', 'fanglan', 'wenhui', 
            'studentcenter', 'dormitory1', 'dormitory2', 'audioback', 'gallery', 
            'doorway']
    counts = []
    percentages = []


    import re

    line = result.readlines()[-2]
    str2 = line.split(' ')
        
    for i in range(20):
        count = 0
        percentage = '0'
        if str2[0] == index[i]:
            count = count + 1
            for j in range(len(str2)-1):
                if '%' in str2[j]:
                    percentage = str2[j]
                    numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', percentage)]
                    str_numbers = str(numbers)
                    str_numbers = str_numbers.replace('[', '')                    
                    str_numbers = str_numbers.replace(']', '')
                    percentage = str_numbers
                    
        counts.append(count)
        percentages.append(percentage)


    import numpy as np 
    import pandas as pd

    df = pd.DataFrame((zip(index1, counts, percentages)), columns = ['class', 'counts', 'accuracy'])
    df = df.set_index('class')
    df1 = df[df['counts'] != 0]
    df1 = df1[df['accuracy'] == max(df['accuracy'])]
    df.to_csv("result.txt", sep='\t', index=False)
    result = list(df1.index.astype(str))[0]

    f = open('../../result.txt', 'w')
    f.writelines(result)
    f.close()

