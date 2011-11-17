#! /usr/bin/perl

$f = `ls -1`;  # get all files in the pwd
@files = split(/\n/,$f);
chomp(@files);

for(@files) {
	if(/prt$/){
		`python ~/Code/mvpa/prt2Labels.py $_ 5 5 STI STU TESTSTI TESTSTU TESTNSU`;
		print("$_\n")
	}
}
