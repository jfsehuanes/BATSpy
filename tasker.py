import sys
import os
import numpy as np

from IPython import embed


def file_filter(folder_path, range):

    valid_files = []
    for filename in os.listdir(folder_path):
        basename, ext = os.path.splitext(filename)
        if ext != '.wav':
            continue
        elif 'channel1' not in basename:
            continue
        try:
            number = int(basename[-3:])
        except ValueError:
            continue  # not numeric
        if int(range[0]) <= number <= int(range[1]):
            # process file
            valid_files.append(filename)

    return valid_files


if __name__ == '__main__':

    # # Task for pc-tape fixer
    # if len(sys.argv) != 3:
    #     print("ERROR\nPlease tell me the folder with your recordings, followed by the folderPath you wish to store the"
    #           " fixed files into.")
    #     quit()
    #
    # folder_path = sys.argv[1]
    # store_path = sys.argv[2]
    #
    # if folder_path[-1] != '/':
    #     folder_path += '/'
    #
    # if store_path[-1] != '/':
    #     store_path += '/'
    #
    # wavfiles = []
    #
    # # Walk through folder_path and search for desired files
    # for root, dir, files in os.walk(folder_path):
    #     for f in files:
    #         if f.endswith('.wav'):
    #             wavfiles.append(os.path.join(root, f))
    #
    # # create parallel jobs for user specific folder:
    # job_name = 'task_list_for_pcTape_fixer.txt'
    # task_list = ['python3 pcTape_file_fixer.py ' + '"%s"' % e + ' ' + store_path for e in wavfiles]
    # task_list = np.sort(task_list)
    #
    # if len(task_list) > 0:
    #     np.savetxt(job_name, task_list, fmt="%s")

    # task for running batspy several times so it stores call parameters
    if len(sys.argv) != 4:
        print("ERROR\nPlease tell me the folder with your recordings, followed by the range of recordings to be "
              "analzed as separate arguments (ex. python tasker.py folder_path 1 17).")
        quit()

    # Extract arguments
    folder_path = sys.argv[1]
    begin = sys.argv[2]
    end = sys.argv[3]
    txt_name = 'call_parameters_extraction_list.txt'

    valid_files = file_filter(folder_path, (begin, end))
    new_tasks = ['python bats.py ' + folder_path + e + ' m\n' for e in valid_files]

    # create the parallel task
    if os.path.exists(txt_name):
        prev_file = open(txt_name, 'a')
        prev_file.writelines(new_tasks)
        prev_file.close()
    else:
        new_file = open(txt_name, 'w')
        new_file.writelines(new_tasks)
        new_file.close()

    # print('Task-list terminated. Tasks stored in %s' % job_name)
