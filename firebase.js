{
  "rules": {
    ".read": "auth != null",

    "$path": {
      ".write": "auth != null && (
        root.child('admins').child(auth.uid).val() === true
        || (!data.exists() && false)
        || (data.exists() && !newData.exists() ? false : true)
      )"
    }
  }
}