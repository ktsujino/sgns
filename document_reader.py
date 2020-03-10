import glob

def document_reader(text_dir, split=True):
    for data_file in glob.glob(text_dir + '/*/*'):
        for line in open(data_file):
            line = line.rstrip('\n')
            if not line.startswith('<doc ') and \
               not line.startswith('</doc>') and \
               line:
                if split:
                    yield line.split()
                else:
                    yield line
                    
