PCPATH=$(/usr/bin/julia -e 'using PackageCompiler; s=joinpath(dirname(dirname(pathof(PackageCompiler))),"juliac.jl"); print(s)')
GRPATH=$(/usr/bin/julia -e 'using GR; s=joinpath(dirname(dirname(pathof(GR))),"deps","gr","lib"); s = s*";"; print(s)')
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GRPATH
julia --startup-file=no $PCPATH --compile all --compiled-modules yes -vare bosched.jl
if [[ $* == *--install* ]]
then
   sudo ln -s $PWD/builddir/bosched /usr/local/bin/bosched
fi
