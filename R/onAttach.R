.onAttach=function(libname,pkgname){
   packageStartupMessage("Loaded adelie ", as.character(packageDescription("adelie")[["Version"]]))
}
