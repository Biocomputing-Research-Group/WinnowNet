import h5py
import sys
import pandas as pd
def get_generic_score(df):
	score = df['hits'] * df['hits']/(2*df['n_frags_db']) + df['hits'] * df['fragments_matched_int_ratio'] + df['hits_y']
	return score.values

if __name__ == "__main__":
	file_path=sys.argv[1]
	hdf_file=h5py.File(file_path,'r')
	dataset=hdf_file['first_search']
	return_dataset_slice=slice(None)
	df = pd.DataFrame({column: dataset[column][return_dataset_slice] for column in sorted(dataset)})
	for column in dataset:
		if df[column].dtype == object:
			df[column] = df[column].apply(lambda x: x if isinstance(x, str) else x.decode('UTF-8'))
	df['generic_score'] = get_generic_score(df)
	new_df=df[['scan_no','sequence','charge','generic_score']]
	new_df['charge']=df['charge'].astype(int)
	new_df.to_csv('wintemp.tsv',sep='\t',index=False)
