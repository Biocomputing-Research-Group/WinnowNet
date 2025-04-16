import h5py
def get_generic_score(df)
    score = df['hits'] * df['hits']/(2*df['n_frags_db']) + df['hits'] * df['fragments_matched_int_ratio'] + df['hits_y']
    return score.values

if __name__ == "__main__":
    hdf_file=h5py.File(file_path,'r')
    df=hdf_file['first_search']
    df['generic_score'] = get_generic_score(df)
    new_df=df[['scan_no','sequence','charge']]
    new_df.to_csv('wintemp.tsv',sep='\t',index=False)
