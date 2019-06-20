
import os
import pickle
import torch
import natsort
import numpy as np
import codecs

train=True
sub=True
mix=True
val=False


if(train):
	root_train ="/mnt/data4/embedding_training_result_sub/train_feat/"
	img_feat_path= os.path.join(root_train, "img_feat")
	feats_train=[]

		#for all the pickle 
	for elt in natsort.natsorted(os.listdir(img_feat_path)):
	  #print("reading ", os.path.join(img_feat_path,elt))
	  with open(os.path.join(img_feat_path, elt), 'rb') as f: #open the pickle
	      feat= torch.from_numpy(pickle.load(f))
	      feats_train.append(feat)
	 
	feats_train=torch.cat(feats_train)
	print(feats_train.shape)
	
	feats_train = feats_train.cpu().data.numpy()
	if(sub):
		feats_train=feats_train[:20000]
		print("reduced to", (len(feats_train)), len(feats_train[0]))
	# for elt in feats:
	#print(feats[0:2])
	np.savetxt(os.path.join(root_train, "feat_train.tsv"), feats_train, delimiter='\t')

	query_synset_dic={}
	with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
	    for line in f:
	        (key, val) = line.split()
	        query_synset_dic[int(key)] = val


	metadata_train=[]
	metadata_train.append(["query", "class"])
	with open(os.path.join(root_train, "img_names_all.lst"), 'r') as f:
		images_names = f.readlines()[1:]
		#print("img, name", images_names[0:2]) 
		for elt in images_names:
			query_id = os.path.dirname(elt)
			query_id = os.path.splitext(query_id)[0].strip('\n') #query id
			query_id=os.path.basename(query_id)
			class_id = int(query_synset_dic[int(query_id[1:])])
			#print("query name", query_id, class_id)
			metadata_train.append([query_id, str(class_id)])
	if sub:
		metadata_train=metadata_train[:20001]
	np.savetxt(os.path.join(root_train, "meta_feat_train.tsv"), metadata_train, fmt="%s", delimiter='\t')



if (mix or val):
	root_val ="/mnt/data4/embedding_training_result_sub/val_feat/"
	img_feat_path= os.path.join(root_val, "img_feat")
	feats_val=[]
	with codecs.open(os.path.join(root_val, "feat_val.tsv"), 'w', 'utf-8') as f:
		for elt in natsort.natsorted(os.listdir(img_feat_path)):
			print("reading ", os.path.join(img_feat_path,elt))
			with open(os.path.join(img_feat_path,elt), 'rb') as f: #open the pickle
				feat= torch.from_numpy(pickle.load(f))
				feats_val.append(feat)
	feats_val=torch.cat(feats_val)
	print(feats_val.shape)
	feats_val = feats_val.cpu().data.numpy()
	# for elt in feats:
	#   print(elt[0])
	#   f.write("\n".join("\t".join(map(str, elt)))) 
	#   f.write("\n".join("\t".join(map(str, elt)))) 
	np.savetxt(os.path.join(root_val, "feat_val.tsv"), feats_val, delimiter='\t')


	val_meta={} #query (key): class
	with open("/mnt/data2/betty/Pictures/webvision/info/val_filelist.txt") as f:
	    for line in f:
	        (key, val) = line.split()
	        val_meta[int(key[3:-4])] = val
	metadata_val=[]
	metadata_val.append(["img_name", "cls_id"])
	with open(os.path.join(root_val, "img_names.lst"), 'r') as f:
		images_names = f.readlines()

		for elt in images_names:
			img_name = os.path.basename(elt)
			img_name = os.path.splitext(img_name)[0].strip('\n') #query id
			#print("name", query_id[3:], int(query_id[3:]))
			class_id = int(val_meta[int(img_name[3:])])
			#print("query name", img_name, class_id)
			metadata_val.append([img_name, str(class_id)])
	print(len(metadata_val))
	np.savetxt(os.path.join(root_val, "meta_feat_val.tsv"), metadata_val, fmt="%s", delimiter='\t')


if (mix):
	metadata_mix = np.concatenate((metadata_train, metadata_val[1:]))
	feat_mix = np.concatenate((feats_train, feats_val))
	print("mix the data...", len(metadata_mix), len(feat_mix))

	np.savetxt(os.path.join(root_train, "feat_mix.tsv"), feat_mix, delimiter='\t')
	np.savetxt(os.path.join(root_train, "meta_feat_mix.tsv"), metadata_mix, fmt="%s", delimiter='\t')


