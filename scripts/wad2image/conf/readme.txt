This directory includes configuration files that can be invoked by the "-c"
command line option. For example, for Yadex mode "-c yadex" tells wad2image to
read "yadex.conf" in this directory. Configuration files are just an
alternative to specifying the same thing on the command line. Configuration
files consist of lines with the form:
    key=value
where "key" is the long form of a wad2image option. Lines starting with "#" and
blank lines are ignored.
