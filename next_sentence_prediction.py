import transformers

def next_sentence_prediction():

    #model = transformers.AutoModelWithLMHead.from_pretrained('C:/Users/zmh001/Documents/language_models/roberta_large')
    #tokenizer = transformers.AutoTokenizer.from_pretrained('C:/Users/zmh001/Documents/language_models/roberta_large/')

    #model = transformers.AutoModelWithLMHead.from_pretrained('C:/Users/zmh001/Documents/language_models/bio_clinical_bert')
    #tokenizer = transformers.AutoTokenizer.from_pretrained('C:/Users/zmh001/Documents/language_models/bio_clinical_bert')

    model = transformers.AutoModelWithLMHead.from_pretrained('C:/Users/zmh001/Documents/language_models/trained_models/bert_pretrained_v2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('C:/Users/zmh001/Documents/language_models/trained_models/bert_new_vocab/')

    nlp_fill = transformers.pipeline('fill-mask', model=model, tokenizer=tokenizer)
    # print(nlp_fill('There is a inflammatory lesion in the patient''s left lung ' + nlp_fill.tokenizer.mask_token))
    # print(type(nlp_fill('Dan is a ' + nlp_fill.tokenizer.mask_token)))

    # new_string = 'the lesion in the patient '
    # new_string = 'the lesion in '
    # new_string = 'i really want '
    # new_string = 'This meeting today '
    #new_string = 'the scan showed '
    #new_string = 'The patients '
    new_string = 'eg. '
    for i in range(0, 10):

        full_return = nlp_fill(new_string + str(nlp_fill.tokenizer.mask_token))
        # print(full_return[0:5])
        # for new_seq in full_return:
        #    seq = new_seq['sequence']
        #    token = new_seq['token_str']
        # if '.' in seq or '!' in seq:
        # full_return.remove(new_seq)

        #    if '.' in token or '!' in token:
        #        full_return.remove(new_seq)

        best_seq = full_return[0]
        for new_seq in full_return:

            if best_seq["token_str"] in '.' or best_seq["token_str"] in '!' or best_seq["token_str"] in '?' or best_seq[
                "token_str"] in '-' or best_seq["token_str"] in ')' or best_seq["token_str"] in ':' or best_seq[
                "token_str"] in ';' or best_seq["token_str"] in '…'  or (best_seq["token_str"] in ' ' and new_string[-1] in ' '):
                best_seq = new_seq

        # print(full_return[0:5])
        return_dict = full_return[0]
        return_token = return_dict["token_str"]
        # if return_token == '.':
        #    return_dict = full_return[1]
        new_string = return_dict['sequence']
        new_string = best_seq['sequence']
        print(new_string)