source ~cs165/tools/loadme.tcl
set e [berkdb env -create -home ./myenv]
set btree [berkdb open -rdonly -env $e /home/c/s/cs165/data/btree.db]
set hash [berkdb open -rdonly -env $e /home/c/s/cs165/data/hash.db]

proc printdb { db } {
	set c [$db cursor]
	for { set pair [$c get -first]} {$pair != ""} {set pair [$c get -next]} {
		puts $pair
	}
}
proc iterate { db } {
	set c [$db cursor]
	for { set pair [$c get -first]} {$pair != ""} {set pair [$c get -next]} {
		set dummyvar $pair
	}
}
proc testperformance { mydb } {
	puts [time {iterate $mydb}]
}
#testperformance $hash
#testperformance $btree

#Question 5:

proc iterate_range { db from to} {
	set c [$db cursor]
	set count 0
	for { set pair [$c get -set $from]} {$pair != ""} {set pair [$c get -next]} {
		set dummyvar $pair
		puts $dummyvar
		incr count
		if  { [lindex [lindex $pair 0] 0] == $to} {
			break
		}
	}
	return $count
}
proc iterate_range_hash { db from to } {
	set c [$db cursor]
	set count 0
	for { set pair [$c get -first]} {$pair != ""} {set pair [$c get -next]} {
		set key [lindex [lindex $pair 0] 0]
		if { [string compare $from $key] <= 0 && [string compare $to $key] >= 0 } {
			incr count
			set dummyvar $pair
			puts $dummyvar
		}
	}
	return $count
}

#puts [iterate_range $btree adcjfiebhg adcjfigbhe]
#puts [iterate_range $btree adcjfiebhg adcjfiegbh]
#puts [iterate_range_hash $hash adcjfiebhg adcjfigbhe]
#puts [iterate_range_hash $hash adcjfiebhg adcjfiegbh]

#Question 9:

proc populate_db { db } {
	for { set i 0 } { $i < 1000000 } { incr i } { 
		$db put $i "this is a value that goes in a key-data pair"
	}
}
proc random_access { mydb } {
	for { set i 0 } { $i < 1000000 } { incr i } {
		set key [expr { int ( rand ( ) * 1000000 ) } ]
		set dummyvar [$mydb get $key]
	}
}
#set hash [berkdb open -create -hash -cachesize {0 2560 0} hash-perf.db]
#set btree [berkdb open -create -btree -cachesize {0 2560 0} btree-perf.db]
#populate_db $hash
#populate_db $btree
#puts [time { random_access $hash }]
#puts [time { random_access $btree }]

#Question 10:

proc file_to_db {infile outdb_name type} {
	set f [open $infile]
	set db [berkdb open -$type -dup -create $outdb_name]
	for { gets $f line } {$line != ""} { gets $f line } {
		set key [keyof $line]
		set data [dataof $line]
		$db put $key $data
	}
	$db close
	close $f
}

#Question 12:

#file_to_db ~cs165/data/rating_to_movie.in rating_to_movie.db btree
#file_to_db ~cs165/data/movie_to_id.in movie_to_id.db btree

#Question 13:
#stuff that I don't think needs to show up in the sql script
#set rtmdb [berkdb open -rdonly ./rating_to_movie.db]
#set mtidb [berkdb open -rdonly ./movie_to_id.db]

$btree close
$hash close
$e close

