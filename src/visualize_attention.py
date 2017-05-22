import numpy as np

input_file = '/Users/diego/Github/SelfAttentiveSentenceEmbedding/output/laptop/attention.txt'

idx = 0
with open(input_file, 'r', encoding='utf-8') as fp:
    for line in fp:
        pred, conf, tokens, attentions = line.rstrip().split('\t')
        pred = int(pred)
        conf = float(conf)
        tokens = tokens.split()
        attentions = np.array([float(a) for a in attentions.split(' ')])
        assert len(attentions) % len(tokens) == 0

        attentions = abs(attentions.reshape(len(tokens), int(len(attentions)/len(tokens))))

        # Plot each row of the attention
        with open('{}.html'.format(idx), 'w', encoding='utf-8') as fp:
            fp.write('<!DOCTYPE html>\n<html>\n<head>\n<title>{0} STARS</title>\n<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>\n<script>\nwindow.onload=function()'.format(pred) + '{\n')
            for i, attention in enumerate(attentions):
                #attention = attention / np.linalg.norm(attention)
                fp.write('var words_{0} = ['.format(i) + '\n')
                res = ''
                for t, a in zip(tokens, attention):
                    res += '{' + "'word': '{}', 'attention': {:.2f}".format(t.replace("\\", "\\\\").replace("'", "\\'").replace("/", "\\/"), 1.0 - a) + '}, '
                fp.write(res[:-2] + '];\n')
                fp.write("$('#text_" + str(i) + "').html($.map(words_" + str(i) + ", function(w) {return '<span style=\"background-color:hsl(360,100%,' + (w.attention * 40 + 60) + '%)\">' + w.word + ' </span>'}))\n")
            fp.write('}\n</script>\n</head>\n<body>\n')
            for i in range(len(attentions)):
                fp.write('<div id="text_{0}"></div><p></p>\n'.format(i))
                for t, a in sorted(zip(tokens, attentions[i]), key=lambda x:-float(x[1])):
                    fp.write('(<b>{}</b>, {}) '.format(t, a))
            fp.write('\n</body>\n</html>')

        attentions = np.log(np.sum(attentions, axis=1))#*100000-100000
        attentions = np.array([x if x >= 0 else 0 for x in attentions])
        attentions = abs(attentions) / np.sum(abs(attentions))
        attentions *= 5
        #attentions = np.log(attentions)
        #attentions = attentions/np.linalg.norm(attentions)
        #attentions = (attentions - min(attentions))/ (max(attentions) - min(attentions))

        with open('{}_all.html'.format(idx), 'w', encoding='utf-8') as fp:
            fp.write('<!DOCTYPE html>\n<html>\n<head>\n<title>{0} STARS</title>\n<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>\n<script>\nwindow.onload=function()'.format(pred) + '{\n')
            fp.write('var words_{0} = ['.format(i) + '\n')
            res = ''
            for t, a in zip(tokens, attentions):
                res += '{' + "'word': '{}', 'attention': {:.2f}".format(t.replace("\\", "\\\\").replace("'", "\\'").replace("/", "\\/"), 1.0 - a) + '}, '
            fp.write(res[:-2] + '];\n')
            fp.write("$('#text_" + str(i) + "').html($.map(words_" + str(i) + ", function(w) {return '<span style=\"background-color:hsl(360,100%,' + (w.attention * 40 + 60) + '%)\">' + w.word + ' </span>'}))\n")
            fp.write('}\n</script>\n</head>\n<body>\n')
            fp.write('<div id="text_{0}"></div><p></p>\n'.format(i))
            for t, a in sorted(list(zip(tokens, attentions)), key=lambda x:-float(x[1])):
                fp.write('(<b>{}</b>, {}) '.format(t, a))
            fp.write('\n</body>\n</html>')

        idx += 1
        if idx > 20:
            break