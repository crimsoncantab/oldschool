<?
    // require common code
    require_once("inc/common.inc");
?>

<!DOCTYPE html 
     PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>Crew Practice Signup</title>
  </head>
  <body>
    <form action="submit.php" method="post">
      <div align="center">
      Athlete:
        <select name="athlete">
          <option selected>Select Name
          <?
              //imports usernames to list in form
              $sql = mysql_query("SELECT username FROM users WHERE name = '" . $_GET["coach"] . "'");
              $row = mysql_fetch_array($sql);
              $_SESSION["name"] = $row["username"];
              $names = mysql_query("SELECT athlete FROM " . $_SESSION["name"]);
              while ($row = mysql_fetch_array($names)) {
                  print("<option>" . $row["athlete"]);
              }
          ?>
        </select>
      </div>
      <br />
      <div align="center">
      Available from:
        <select name="pretimeh">
          <option selected>12
          <option>1
          <option>2
          <option>3
          <option>4
          <option>5
          <option>6
          <option>7
        </select>
        :
        <select name="pretimem">
          <option selected>00
          <option>15
          <option>30
          <option>45
        </select>
      </div>
      <br />
      <div align="center">
      Until:
        <select name="posttimeh">
          <option>12
          <option>1
          <option>2
          <option>3
          <option>4
          <option>5
          <option>6
          <option selected>7
        </select>
        :
        <select name="posttimem">
          <option selected>00
          <option>15
          <option>30
          <option>45
        </select>
      </div>
      <br />
      <div align="center">
      <input type="submit" value="Submit" />
      </div>
    </form>
  </body>
</html>
