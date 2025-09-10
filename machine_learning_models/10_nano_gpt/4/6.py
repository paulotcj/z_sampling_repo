import argparse

'''
Example usage: py 6.py --argword oluap --qwert 112233 -t something
'''

#-------------------------------------------------------------------------
def main():
    print('hello from main')
    # Create the parser
    parser = argparse.ArgumentParser(description='script description')

    # Add arguments
    parser.add_argument('--argword', '-a',                        type=str, help='argword')
    parser.add_argument('--qwert',   '-q',   type=int, default=1, help='int test arg')
    parser.add_argument('--tko',     '-t',                      help='var, it can be anything')

    # Parse arguments
    args = parser.parse_args()

    if args.argword:
        print(f'args.argword, {args.argword}')
    # if args.a: #error
    #     print(f'args.a, {args.a}')
    if args.qwert:
        print(f'args.qwert, {args.qwert}!')
    if args.tko:
        print(f'args.tko, {args.tko}!')
#-------------------------------------------------------------------------


if __name__ == "__main__":
    main()