#!/usr/bin/perl

use List::Util qw(first max maxstr min);

$CNFHOME=".";

require "$CNFHOME/RaptorX-bin/BioDSSP.pm";

#check arguments
(@ARGV >= 1) or die <<"VEND";
Usage: run_raptorx-ss8.pl [sequence file]
sequence file should be a one-line file containing a peptide sequence.
VEND

#Deal with the filename of sequence.
@paths=split /\//,$ARGV[0];
$pathsnum=scalar(@paths)-1;
my @tmp=split /\./,$paths[$pathsnum]; #./
my $numtmp=scalar(@tmp)-2;
$numtmp = 0 if($numtmp<0);
$seqPrefix=join(".",@tmp[0..$numtmp]) . '_raptorx';

$jobtmpdir=`$CNFHOME/RaptorX-bin/mktemp -d --tmpdir=./ --suffix=.$seqPrefix`;
chomp($jobtmpdir);
$jobid=int(rand(99999999));
$jobid="$seqPrefix.$jobid";
$FNTMPSEQ="$jobtmpdir/cnfsseight.$jobid.seq";
$FNTINPUT="$jobtmpdir/cnfsseight.$jobid.input";
$FNTMPPSSM=$ARGV[0];##shj 171115


$seq = "";
my @tmprst=ParsePSSM("$ARGV[0]");
my @seq=@{$tmprst[0]};
$seq = join("",@seq);
print "seq[$seq]\n"; 


open FTMPSEQ,">$FNTMPSEQ";
print FTMPSEQ "$seq\n";
close FTMPSEQ;


print STDERR "Generating features...\n";
$tmp=`$CNFHOME/RaptorX-bin/GenFeat_nodssp.pl $FNTMPSEQ $FNTMPPSSM $CNFHOME/RaptorX-data/p.p2.c.unit $FNTINPUT $CNFHOME`;
$FNTMPRST="$jobtmpdir/cnfsseight.$jobid.result";
$FNTMPNULL="$jobtmpdir/cnfsseight.$jobid.null";
open FTMPNULL,">$FNTMPNULL";
print FTMPNULL "0\n";
close FTMPNULL;
print STDERR "Predicting 8-class secondary structure using CNF model...\n";
$cmd="$CNFHOME/RaptorX-bin/bcnf_mpitp CONF $CNFHOME/RaptorX-data/CNF.ax.norm.conf ACT null PREDICT1 1 FEATMASK 0 ALLRST all.rst.csv BIEN 1 TRAIN $FNTMPNULL TEST $FNTINPUT RESULT $FNTMPRST RESUME $CNFHOME/RaptorX-data/model.p0-900.1561189 &> $jobtmpdir/cnfsseight.$jobid.stderr";
`$cmd`;

#Parse the output of cnf ss eight prediction
open fhCNFOutput,"<$FNTMPRST" or die $!;
$rstline=<fhCNFOutput>;
$rstline=~s/^\s+//;
chomp($rstline);
@p=split/\s+/, $rstline;
my @alllab=split //, $p[6];
my @prob=@p[7..scalar(@p)-1];
close fhCNFOutput;

@alllab=convert_label_to_letter(@alllab);
#Output the data in letter form, each line is 
# [number] [amino acid] [ss in eight letters]  [eight probability]
print STDERR "Formating results...\n";


open fhRESULT,">$seqPrefix.ss8";
@allseq=split //, $seq;
print fhRESULT "#RaptorX-SS8: eight-class secondary structure prediction results\n";
print fhRESULT "#probabilities are in the order of H G I E B T S L(loops), the 8 secondary structure types used in DSSP\n\n";
for($i=0;$i<@allseq;$i++)
{
    
    print fhRESULT sprintf("%4d %s %s   ",$i+1,$allseq[$i], $alllab[$i]);
#    print fhRESULT "$allseq[$i]\t $alllab[$i]\t";
    for($k=0;$k<8;$k++)
    {
	print fhRESULT sprintf("%.3f ",$prob[$i*8+$k]);
#	print fhRESULT $prob[$i*8+$k],"\t";
    } 
    print fhRESULT "\n";
} 
close fhRESULT;
print STDERR "DONE.\n";
`rm -rf output*`;
if(!defined $ARGV[1]){
`rm cnfsseight.$jobid.*`;
}

#=============predict 3-class secondary structure==============
print STDERR "Predicting 3-class secondary structure using CNF model...\n";
$cmd="$CNFHOME/RaptorX-bin/bcnf_mpitp CONF $CNFHOME/RaptorX-data/3state.conf ACT null PREDICT1 1 FEATMASK 0 ALLRST all.rst.csv BIEN 0 TRAIN $FNTMPNULL TEST $FNTINPUT RESULT $FNTMPRST RESUME $CNFHOME/RaptorX-data/3state-model.p0-1300.1342925 &> cnfsseight.$jobid.stderr";
`$cmd`;

#Parse the output of cnf ss eight prediction
open fhCNFOutput,"<$FNTMPRST" or die $!;
my $horiz;
my @hoConf=qw();

$rstline=<fhCNFOutput>;
$rstline=~s/^\s+//;
chomp($rstline);
@p=split/\s+/, $rstline;
my @alllab=split //, $p[6];
my @prob=@p[7..scalar(@p)-1];
close fhCNFOutput;

@alllab=convert_label_to_letter3(@alllab);
#Output the data in letter form, each line is 
# [number] [amino acid] [ss in eight letters]  [eight probability]
print STDERR "Formating results...\n";


open fhRESULT,">$seqPrefix.ss3";
@allseq=split //, $seq;
print fhRESULT "#RaptorX-SS3: three-class secondary structure prediction results\n";
print fhRESULT "#probabilities are in the order of H(alpha helix) E(beta strand) C(loops), the 3 secondary structure types used in PSIPRED\n\n";
for($i=0;$i<@allseq;$i++)
{
    
    print fhRESULT sprintf("%4d %s %s   ",$i+1,$allseq[$i], $alllab[$i]);
    for($k=0;$k<3;$k++)
    {
	print fhRESULT sprintf("%.3f ",$prob[$i*3+$k]);
    } 
    print fhRESULT "\n";
    push @hoConf,sprintf("%1d",int(10*max(@prob[$i*3..$i*3+2])-1));
} 
close fhRESULT;

open fhHO,">$seqPrefix.horiz";

print fhHO "#RaptorX-SS3: three-class secondary structure prediction results\n\n";

for(my $i=0;$i<@allseq;$i=$i+60)
{
    my $end;
    if($i+60>@allseq)
    {
	$end=@allseq-1;
    }
    else{
	$end=$i+60-1;
    }
    print fhHO "Conf: ",join("",@hoConf[$i..$end]),"\n";
    print fhHO "Pred: ",join("",@alllab[$i..$end]),"\n";
    print fhHO "  AA: ",join("",@allseq[$i..$end]),"\n";
    print fhHO "      ";
    for(my $j=$i+10;$j<=$end+1;$j=$j+10){
	print fhHO sprintf("%10d",$j);
    }
    print fhHO "\n"x3;
}
close fhHO;


print STDERR "DONE.\n";
`rm -rf output*`;
if(!defined $ARGV[1]){
`rm cnfsseight.$jobid.*`;
}

