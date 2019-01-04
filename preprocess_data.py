import os 

top_dir = "orl_faces"

def statistic_data(top_dir):

    subjects_dir = [i for i in os.listdir(top_dir) if os.path.isdir(top_dir+"/"+i)]
    print len(subjects_dir)
    exams = [] 
    for d in subjects_dir:
        exams.append(len(os.listdir(top_dir+"/"+d)))
    for d, n in zip(subjects_dir, exams):
        print "{} : {}".format(d, n)


if __name__ == "__main__":
    statistic_data(top_dir)

