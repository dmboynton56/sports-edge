import nflreadpy as nfl

def main():
    pbp = nfl.load_pbp()
    pbp_pandas = pbp.to_pandas()
    print(pbp_pandas.head())

if __name__ == "__main__":
    main()