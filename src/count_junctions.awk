BEGIN {
	read=0;
	junctions=0;
}
{
	if (read==1) {
		if (NF!="1") {
			junctions++;
		}
		else {
			read=0;
		}
	}
	if ($1~"JUNCTIONS") {
		read=1;
	}
}
END {
	OFS=",";
	print FILENAME, junctions-1;
}
