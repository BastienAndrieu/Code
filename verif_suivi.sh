# Résultat de 'git status' sans les fichiers temporaires '*.~'

list="$(git status)"
IFS=$(echo -en "\n\b")
ary=($list)
for key in "${!ary[@]}"
do 
    str="${ary[$key]}"
    last="${str: -1}"
    if [ "$last" != "~" ]
    then
	echo "$str"
    fi
	
done

