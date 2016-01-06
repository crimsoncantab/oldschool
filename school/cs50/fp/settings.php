<?
    // require common code
    require_once("inc/common.inc");
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
               "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Coach's Site</title>
  </head>
  <body>
    <form action="generate.php" method="get">
      <div>
        <input type="checkbox" name="side" value="true" checked />
        Account for starboard/port preference?
      </div>
      <div>
        <input type="checkbox" name="size" value="true" checked />
        Create boats of four?
      </div>
      <div>
        Maximum boats per practice:
        <input type="textbox" name="boatmax" value="2"/>
      </div>
      <div>
        Length of practices (in hours):
        <input type="textbox" name="length" value="1.5"/>
      </div>
      <div>
        Minimum time between practices (in hours):
        <input type="textbox" name="interim" value="1"/>
      </div>
      <div>
      Latest Practice:
        <select name="lateh">
          <option>1
          <option>2
          <option>3
          <option>4
          <option selected>5
          <option>6
          <option>7
        </select>
        :
        <select name="latem">
          <option>00
          <option>15
          <option selected>30
          <option>45
        </select>
      </div>
      <div>
        <input type="submit" value="Generate Practice Times" />
      </div>
    </form>
    <br />
    <br />
    <form action="add.php" method="post">
      <div>
        <input type="textbox" name="athlete" value="Player Name" />
      </div>
      <div>
        <input type="radio" name="pref" value="p" checked />Port<br />
        <input type="radio" name="pref" value="s" />Starboard<br />
        <input type="radio" name="pref" value="c" />Coxswain<br />
      <div>
      <div>
        <input type="submit" value="Add Player" />
      </div>
    </form>
    <br />
    <br />
    <form action="delete.php" method="post">
      <div>
        <select name="athlete">
          <option selected>
            <?
                //imports athletes from database to be listed in form
                $names = mysql_query("SELECT athlete FROM " . $_SESSION["username"]);
                while ($row = mysql_fetch_array($names)) {
                    print("<option>" . $row["athlete"]);
                }
            ?>
        </select>
      </div>
      <div>
        <input type="submit" value="Remove Player" />
      </div>
    </form>
    <br />
    <br />
    <div>
      <a href="logout.php">Log Out</a>
    </div>
</html>
