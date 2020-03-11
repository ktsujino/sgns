from gensim import downloader
import glob

def document_reader(text_dir, split=True):
    if not text_dir:
        text8 = downloader.load('text8')
        for line in text8:
            if split:
                yield line
            else:
                yield ' '.join(line)
    else:
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
                    
