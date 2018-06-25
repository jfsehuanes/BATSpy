import sys
import os
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("ERROR\nPlease tell me the folder with your recordings, followed by the folderPath you wish to store the fixed files into.")
        quit()

    folder_path = sys.argv[1]
    store_path = sys.argv[2]

    if folder_path[-1] != '/':
        folder_path += '/'

    if store_path[-1] != '/':
        store_path += '/'

    wavfiles = []

    # Walk through folder_path and search for desired files
    for root, dir, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.wav'):
                wavfiles.append(os.path.join(root, f))

    # create parallel jobs for user specific folder:
    job_name = 'task_list_for_pcTape_fixer.txt'
    task_list = ['python3 pcTape_file_fixer.py ' + '"%s"' % e + ' ' + store_path for e in wavfiles]
    task_list = np.sort(task_list)

    if len(task_list) > 0:
        np.savetxt(job_name, task_list, fmt="%s")

    print('Task-list terminated. Tasks stored in %s' % job_name)
