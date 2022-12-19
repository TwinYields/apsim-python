
# Python interface to APSIM next generation

## Installation

First install APSIM and add the directory containing the Models executable to path (to find the right .dll files). On Windows you can install the APSIM binary. 

On Linux you can use the following to build APSIM: 

```bash
git clone --depth 1 https://github.com/APSIMInitiative/ApsimX.git
dotnet build -o ~/.local/lib/apsimx -c Release ApsimX/Models/Models.csproj
```

And add to Pythonpath:

```bash
export PYTHONPATH=~/.local/lib/apsimx
```
